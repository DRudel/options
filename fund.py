from fund_model import FundModel
from features_v2 import generate_full_data, GROWTH_DICT, GROWTH_NAMES, NON_FEATURES

# from features import prepare_data, GROWTH_DICT, GROWTH_NAMES, VOLATILITY_LIST, \
#     calc_avg_abs_change, NON_FEATURES, PRICING_VOLATILITIES
from pricing_models import NormPricer
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from utilities import calculate_prices, calc_final_value, form_pricing_data
from numpy.random import default_rng
from result_evaluator import RegressionEvaluator
from copy import deepcopy

my_rng = default_rng()

MIN_MARGIN_EQUIVALENT = 10
# MAX_MARGIN_EQUIVALENT = 3.5
MAX_MARGIN_EQUIVALENT = 8
EVALUATION_EQUIVALENT = 0.75

def calc_avg_abs_monthly_change(my_series):
    ratios = 100 * (my_series - my_series.shift(20)) / my_series.shift(20)
    return ratios.abs().mean()


def select_by_integer_index(df, selection, keep=True):
    idx = np.ones(len(df.index), dtype=bool)
    idx[selection] = False
    if keep:
        idx = ~idx
    return df.iloc[idx]


class Fund:

    @staticmethod
    def extend_fund(old_fund: 'Fund', **kwargs):
        new_fund = Fund(**kwargs)
        new_fund.pricing_model = old_fund.pricing_model
        new_fund.set_growth_data()
        for (num_days, this_margin) in old_fund.models:
            fund_model = FundModel(new_fund.full_data, margin=this_margin, num_days=num_days,
                                   feature_indexes=new_fund.feature_indexes, pricing_model=new_fund.pricing_model)
            print(num_days, this_margin)
            fund_model.assign_labels()
            fund_model.features_to_use = old_fund.models[(num_days, this_margin)].features_to_use
            new_fund.models[(num_days, this_margin)] = fund_model
        return new_fund

    def __init__(self, name: str, base_data: pd.DataFrame, feature_indexes=None, feature_prep=None, call=True,
                 subsample=None):
        if feature_prep is None:
            feature_prep = generate_full_data
        self.name = name
        self.full_data: pd.DataFrame = feature_prep(base_data)
        self._working_data = None
        self.volatility_features = [x for x in self.full_data.columns if 'vol' in x]
        self.feature_indexes = feature_indexes
        self.models: dict[tuple, FundModel] = dict()
        self.average_volatility = calc_avg_abs_monthly_change(base_data['close'])
        self.evaluation_margin_dict = dict()
        self.margin_dict = dict()
        self.pricing_model = None
        self.feature_processor = feature_prep
        self.set_feature_indexes()
        self.pricing_vol = None
        self.growth_data= pd.DataFrame()
        self.call = call
        for num_days in GROWTH_NAMES:
            tp_vol = np.sqrt(num_days) * self.average_volatility
            min_margin = int(np.floor(tp_vol / MIN_MARGIN_EQUIVALENT))
            max_margin = int(np.ceil(tp_vol / MAX_MARGIN_EQUIVALENT))
            self.margin_dict[num_days] = range(min_margin, max_margin + 1)


    def reset_working_data(self, frac, reset_models=True):
        self._working_data = self.full_data.sample(frac=frac, replace=False)
        self._working_data.sort_index(inplace=True)
        if reset_models:
            for model in self.models.values():
                model.raw_data = self._working_data
                model.assign_labels()

    def generate_growth_data(self, vol_name, frac=None):
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=False)
        growth_data_chunks = []
        for gn in GROWTH_DICT:
            growth_data_chunks.append(form_pricing_data(self._working_data, gn, vol_name))
        growth_data = pd.concat(growth_data_chunks, axis=0)
        growth_data.dropna(inplace=True)
        return growth_data

    def set_growth_data(self, frac=None):
        assert self.pricing_model is not None, 'set_growth_data called before pricing vol set'
        self.growth_data = self.generate_growth_data(self.pricing_model.vol_name, frac=frac)

    def predict(self, input_df: pd.DataFrame, price_today=None, num_days_offset=0):
        full_data = self.feature_processor(input_df)
        output_df = full_data.copy()
        full_data = full_data.fillna(method='ffill')
        full_data = full_data.dropna()
        dates = []
        price_categories = []
        months = []
        num_days = []
        margins = []
        strike_prices = []
        option_prices = []
        probabilities = []
        estimated_profits = []
        model_values = []
        model_advantages = []
        model_complexities = []
        pricing_offsets = []
        for scenario in self.models:
            scenario_model = self.models[scenario]
            results = scenario_model.predict_outcomes(full_data)
            to_join = pd.DataFrame(
                {
                    'scenario_' + str(scenario) + '_prob': results['prob'],
                    'scenario_' + str(scenario) + '_prob_expensive': results['prob_dear'],
                    'scenario_' + str(scenario) + '_prob_cheap': results['prob_cheap'],
                    'scenario_' + str(scenario) + '_price': results['assumed_price']
                }, index=results.index
            )
            output_df = output_df.join(to_join)
            final_row = full_data.iloc[-1:, :].copy()
            last_input_price = final_row['close'].values[0]
            if price_today is None:
                final_price = last_input_price
            else:
                final_price = price_today
            margin = scenario[1]
            dates.extend(3 * [final_row.index[0]])
            price_categories.extend(['cheap', 'medium', 'expensive'])
            months.extend(3 * [int(scenario[0] / 30)])
            margins.extend(3 * [margin])
            model_values.extend(3 * [scenario_model.trained_value])
            model_advantages.extend(3 * [scenario_model.trained_advantage])
            model_complexities.extend(3 * [len(scenario_model.features_to_use)])
            strike_prices.extend(3 * [(1 + margin/100) * final_price])
            final_results = scenario_model.predict_outcomes(final_row, num_days_offset=num_days_offset)
            num_days.extend(3 * [final_results['num_days'].values[0]])
            pricing_offsets.extend(3 * [final_results['pricing_offset'].values[0]])
            assumed_relative_price = final_results['assumed_price'].values[0]
            prices = [final_price * (assumed_relative_price * scenario_model.cheap_price_modifier) / 100,
                                  final_price * assumed_relative_price / 100,
                           final_price * (assumed_relative_price * scenario_model.dear_price_modifier) / 100]
            option_prices.extend(prices)
            probabilities.extend([final_results['prob_cheap'].values[0], final_results['prob'].values[0],
                                final_results['prob_dear'].values[0]])
            price_differences = [final_price * assumed_relative_price / 100 - x for x in prices]
            these_profits = [final_results['estimated_profit'].values[0] - x for x in price_differences]
            estimated_profits.extend(these_profits)
        recommendations = pd.DataFrame(
            {
                'date': dates,
                'category': price_categories,
                'num_months': months,
                'num_days': num_days,
                'margin': margins,
                'strike price': strike_prices,
                'option price': option_prices,
                'price offset': pricing_offsets,
                'probability': probabilities,
                'estimated profit': estimated_profits,
                'model advantage': model_advantages,
                'model values': model_values,
                'model complexity': model_complexities
            }
        )
        return output_df, recommendations

    def create_models(self, num_days, margins=None, master_seed=None, overwrite=False, frac=None, **kwargs):
        if margins is None:
            margins = self.margin_dict[num_days]
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=False)
        else:
            self._working_data = self.full_data
        for this_margin in margins:
            if (num_days, this_margin) in self.models:
                if not overwrite:
                    continue
            print(f'training model for num_days = {num_days} and margin = {this_margin}')
            fund_model = FundModel(self._working_data, margin=this_margin, num_days=num_days,
                                   feature_indexes=self.feature_indexes, pricing_model=self.pricing_model)
            fund_model.assign_labels()
            fund_model.select_features(master_seed=master_seed, **kwargs)
            self.models[(num_days, this_margin)] = fund_model
            self.save('post_' + str(num_days) + '_' + str(this_margin))

    # def refresh(self, data, frac=None):
    #     if frac is not None:
    #         self.reset_working_data(frac=frac, reset_models=True)
    #     else:
    #         self._working_data = self.full_data
    #     for fund_model in self.models.values():
    #         fund_model.raw_data = data
    #         fund_model.assign_labels()

    def train_classifiers(self, frac=None, **kwargs):
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=True)
        else:
            self._working_data = self.full_data
        for key in self.models:
            this_fund_model: FundModel = self.models[key]
            this_fund_model.train_classifier(**kwargs)

    def train_regressors(self, frac=None, **kwargs):
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=True)
        else:
            self._working_data = self.full_data
        for key in self.models:
            this_fund_model: FundModel = self.models[key]
            this_fund_model.train_regressor(**kwargs)

    def set_pricing_model(self, price_model_prototype, thresholds, volatilities_to_check=None, rough=False,
                          frac=None):
        if volatilities_to_check is None:
            volatilities_to_check = self.volatility_features
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=False)
        else:
            self._working_data = self.full_data
        for vn in volatilities_to_check:
            assert vn in self._working_data.columns, f'{vn} not in data'
        best_error = None
        best_pricing_model = None
        for volatility_name in volatilities_to_check:
            print(f'Checking volatility {volatility_name}')
            growth_data = self.generate_growth_data(volatility_name)
            this_pricing_model = deepcopy(price_model_prototype)
            this_pricing_model.vol_name = volatility_name
            this_error = this_pricing_model.train(data=growth_data, thresholds=thresholds, return_loss=True,
                                                  rough=rough)
            print()
            print(f'error: {this_error}; best error seen earlier: {best_error}')
            if best_error is None or this_error < best_error:
                best_error = this_error
                best_pricing_model = this_pricing_model
        self.pricing_model: NormPricer = best_pricing_model
        print(f'Volume feature {self.pricing_model.vol_name} selected for pricing.')
        self.set_growth_data()

    def set_feature_indexes(self):
        feature_indexes = list(range(len(self.full_data.columns)))
        for nf in NON_FEATURES:
            feature_indexes.remove(list(self.full_data.columns).index(nf))
        self.feature_indexes = feature_indexes

    def save(self, memo: str = None):
        if memo is None:
            memo = str(datetime.now())[:19]
        memo = memo.replace(':', '_')
        pickle.dump(self, open('model_snapshots/' + self.name + '_' + memo + '.pickle', 'wb'))

    def report_features_used(self):
        features_used = []
        data: pd.DataFrame = self.full_data.iloc[:, self.feature_indexes]
        # data.info()
        for model in self.models.values():
            feature_names = [data.columns[int(x)] for x in model.features_to_use]
            features_used.extend(feature_names)
        idx_df = pd.DataFrame({
            'indexes': features_used
        })
        feature_names = data.columns.to_frame()
        counts = idx_df['indexes'].value_counts().to_frame()
        report = feature_names.join(counts)
        report = report[['indexes']]
        report.sort_values('indexes', inplace=True, ascending=False)
        report = report.fillna(0)
        print(report)

    def tune_classifiers(self, evaluator, overwrite=False, frac=None):
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=True)
        else:
            self._working_data = self.full_data
        for fund_model_key in self.models:
            print(fund_model_key)
            fund_model: FundModel = self.models[fund_model_key]
            if fund_model.num_leaves_classification is not None:
                if not overwrite:
                    continue
            num_classification_leaves, best_summary = \
                fund_model.select_leaf_count(model=fund_model.classification_model, classification=True,
                 data_provider=fund_model.classification_data_provider, jitter_count=7, num_selection_bundles=20,
                 cont_jitter_magnitude=0.20, results_evaluator=evaluator)
            fund_model.num_leaves_classification = num_classification_leaves
            fund_model.trained_advantage = best_summary['advantage'].iloc[0]
            fund_model.trained_value = best_summary['value'].iloc[0]
            # fund_model.tune_model(num_selection_bundles=10, results_evaluator=evaluator, jitter_count=7,
            #                       cont_jitter_magnitude=0.15)
            self.save('classifiers_tuned')

    def tune_regressors(self, overwrite=False, frac=None):
        if frac is not None:
            self.reset_working_data(frac=frac, reset_models=True)
        else:
            self._working_data = self.full_data
        for fund_model_key in self.models:
            print(fund_model_key)
            fund_model: FundModel = self.models[fund_model_key]
            if fund_model.num_leaves_regression is not None:
                if not overwrite:
                    continue
            # regression_features = fund_model.features_to_use + [fund_model.price_idx]
            num_regression_leaves, best_summary = \
                fund_model.select_leaf_count(model=fund_model.regression_model, classification=False,
                                             data_provider=fund_model.regression_data_provider, jitter_count=7,
                                             num_selection_bundles=20, cont_jitter_magnitude=0.10,
                                             results_evaluator=RegressionEvaluator())
            fund_model.num_leaves_regression = num_regression_leaves
            self.save('regression_tuned')