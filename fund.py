from fund_model import FundModel
from features import prepare_data, GROWTH_DICT, GROWTH_NAMES, VOLATILITY_LIST, \
    calc_avg_abs_change, NON_FEATURES, PRICING_VOLATILITIES
from pricing_models import NormCallPricer
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from utilities import calculate_prices, calc_final_value
from numpy.random import default_rng
from scipy import stats
from scipy.optimize import minimize
import sklearn.tree as tree
from ffs.data_provider import ComboDataProvider

my_rng = default_rng()

MIN_MARGIN_EQUIVALENT = 1.3
MAX_MARGIN_EQUIVALENT = 0.5
EVALUATION_EQUIVALENT = 0.75


def evaluate_factor(factors, df, margin, num_months, vol_name, change_name, penalize_bias=True):
    clean_data = df.copy().dropna(subset=[vol_name, change_name])
    base_factor = factors[1]
    prices = 100 * clean_data.apply(calculate_prices, axis=1, margin=margin, num_months=num_months,
                                    vol_name=vol_name, vol_factor=factors[0], base_factor=base_factor)
    values = calc_final_value(clean_data[change_name], margin)
    diff = prices - values
    squared_error = diff * diff
    if penalize_bias:
        return abs(diff.mean()) + squared_error.mean()
    print(f'mse = {squared_error.mean()}; bias = {diff.mean()}')
    return squared_error.mean(), diff.mean()


def select_by_integer_index(df, selection, keep=True):
    idx = np.ones(len(df.index), dtype=bool)
    idx[selection] = False
    if keep:
        idx = ~idx
    return df.iloc[idx]


class Fund:

    def __init__(self, name: str, base_data: pd.DataFrame, feature_indexes=None, feature_prep=None):
        if feature_prep is None:
            feature_prep = prepare_data
        self.name = name
        self.data: pd.DataFrame = feature_prep(base_data)
        self.feature_indexes = feature_indexes
        self.models: dict[tuple, FundModel] = dict()
        self.average_volatility = calc_avg_abs_change(base_data['price'], 1).mean()
        # self.eval_volatility_dict = dict()
        self.evaluation_margin_dict = dict()
        self.margin_dict = dict()
        self.pricing_models = dict()
        #self.vol_factor_dict = dict()
        self.feature_processor = feature_prep
        self.set_feature_indexes()
        for num_months in GROWTH_NAMES:
            tp_vol = np.sqrt(num_months) * self.average_volatility
            min_margin = int(np.floor(tp_vol / MIN_MARGIN_EQUIVALENT))
            max_margin = int(np.ceil(tp_vol / MAX_MARGIN_EQUIVALENT))
            self.margin_dict[num_months] = range(min_margin, max_margin + 1)
            # self.evaluation_margin_dict[num_months] = np.round(tp_vol / EVALUATION_EQUIVALENT)

    def predict(self, input_df: pd.DataFrame, price_today=None):
        full_data = self.feature_processor(input_df)
        output_df = full_data.copy()
        full_data = full_data.fillna(method='ffill')
        full_data = full_data.dropna()
        dates = []
        price_categories = []
        months = []
        margins = []
        strike_prices = []
        option_prices = []
        predictions = []
        model_values = []
        model_advantages = []
        model_complexities = []
        for scenario in self.models:
            scenario_model = self.models[scenario]
            results = scenario_model.predict_outcomes(full_data)
            to_join = pd.DataFrame(
                {
                    'scenario_' + str(scenario) + '_prob': results['outcome'],
                    'scenario_' + str(scenario) + '_prob_plus': results['outcome_plus'],
                    'scenario_' + str(scenario) + '_prob_minus': results['outcome_minus'],
                    'scenario_' + str(scenario) + '_price': results['assumed_price']
                }, index=results.index
            )
            output_df = output_df.join(to_join)
            final_row = full_data.iloc[-1:, :].copy()
            last_input_price = final_row['price'].values[0]
            modifer = 1 # For converting between index value and fund value
            if price_today is None:
                final_price = last_input_price
            else:
                final_price = price_today
                modifier = price_today / last_input_price
            margin = scenario[1]
            dates.extend(3 * [final_row['date'].values[0]])
            price_categories.extend(['cheap', 'medium', 'expensive'])
            months.extend(3 * [scenario[0]])
            margins.extend(3 * [margin])
            model_values.extend(3 * [scenario_model.trained_value])
            model_advantages.extend(3 * [scenario_model.trained_advantage])
            model_complexities.extend(3 * [len(scenario_model.features_to_use)])
            strike_prices.extend(3 * [(1 + margin/100) * final_price])
            final_results = scenario_model.predict_outcomes(final_row)
            assumed_price = final_results['assumed_price'].values[0]
            option_prices.extend([final_price * (assumed_price - 0.5) / 100, final_price * assumed_price / 100,
                           final_price * (assumed_price + 0.5) / 100])
            predictions.extend([final_results['outcome_minus'].values[0], final_results['outcome'].values[0],
                                final_results['outcome_plus'].values[0]])
        recommendations = pd.DataFrame(
            {
                'date': dates,
                'category': price_categories,
                'num_months': months,
                'margin': margins,
                'strike price': strike_prices,
                'option price': option_prices,
                'probability': predictions,
                'model advantage': model_advantages,
                'model values': model_values,
                'model complexity': model_complexities
            }
        )
        return output_df, recommendations

    def create_models(self, num_months, overwrite=False, **kwargs):
        #pricing_vol = self.eval_volatility_dict[GROWTH_NAMES[num_months]]
        pricing_model = self.pricing_models[num_months]
        for this_margin in self.margin_dict[num_months]:
            if (num_months, this_margin) in self.models:
                if not overwrite:
                    continue
            print(f'training model for num_months = {num_months} and margin = {this_margin}')
            #vol_factors = self.vol_factor_dict[(num_months, this_margin)]
            fund_model = FundModel(self.data, margin=this_margin, num_months=num_months,
                                   feature_indexes=self.feature_indexes, pricing_model=pricing_model)
            fund_model.assign_labels()
            training_history = fund_model.select_features(**kwargs)
            self.models[(num_months, this_margin)] = fund_model
            pickle.dump(self, open(self.name + '_post_' + str(num_months) + '_' + str(this_margin) + '.pickle', 'wb'))

    def train_models(self, **kwargs):
        for key in self.models:
            (num_months, margin) = key
            this_fund_model: FundModel = self.models[key]
            this_fund_model.train(**kwargs)

    def assign_pricing_models(self, volatilities_to_check=None, time_periods=None):
        if time_periods is None:
            time_periods = GROWTH_NAMES
        for num_months in time_periods:
            self.assign_pricing_model(num_months, volatilities_to_check)
            self.save()

    def assign_pricing_model(self, num_months, volatilities_to_check):
        assert num_months in list(GROWTH_NAMES.keys()), "time period not in growth dictionary"
        time_period = GROWTH_NAMES[num_months]
        print()
        print('Checking time period ', str(num_months))
        growths = self.data[time_period]
        if volatilities_to_check is None:
            volatilities_to_check = PRICING_VOLATILITIES
        best_error = None
        best_pricing_model = None
        for volatility_name in volatilities_to_check:
            print(f'Checking volatility {volatility_name}')
            this_df = pd.DataFrame({
                'growth': growths,
                'vol': self.data[volatility_name]
            })
            this_df.dropna(inplace=True)
            this_pricing_model = NormCallPricer()
            thresholds = list(self.margin_dict[num_months])
            this_error = this_pricing_model.train(data=this_df, thresholds=thresholds, return_loss=True)
            # this_error = this_pricing_model.train(this_df['growth'], this_df['vol'], thresholds=thresholds,
            #                                       volatility_name=volatility_name, return_loss=True,
            #                                       prototype=best_pricing_model)
            print()
            print(f'error: {this_error}')
            if best_error is None or this_error < best_error:
                best_error = this_error
                best_pricing_model = this_pricing_model
        self.pricing_models[num_months] = best_pricing_model

    def set_feature_indexes(self):
        feature_indexes = list(range(len(self.data.columns)))
        for nf in NON_FEATURES:
            feature_indexes.remove(list(self.data.columns).index(nf))
        self.feature_indexes = feature_indexes

    def save(self, memo:str = None):
        if memo is None:
            memo = str(datetime.now())[:19]
        memo = memo.replace(':', '_')
        pickle.dump(self, open(self.name + '_' + memo + '.pickle', 'wb'))

    def report_features_used(self):
        features_used = []
        data: pd.DataFrame = self.data.iloc[:, self.feature_indexes]
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


