import pandas as pd
import numpy as np
from pricing import euro_vanilla
from features import GROWTH_NAMES
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import pickle
from utilities import calc_final_value, form_growth_df
from ffs.feature_evaluation import SingleFeatureEvaluationRound
from ffs.data_provider import ComboDataProvider
from ffs.train_test import TimeSeriesSplitter, BasicBundleProvider
from ffs.jitter import JitterSetGen
from result_evaluator import ResultEvaluator
from datetime import datetime
from pricing_models import NormCallPricer


DEFAULT_MODEL_PROTOTYPE = GradientBoostingClassifier(max_depth=3, init='zero', n_estimators=7)


class FundModel:

    @staticmethod
    def translate_labels_and_weights(labels, weights, translation):
        profits = labels * weights
        new_profits = profits + translation
        new_labels = np.sign(new_profits)
        new_weights = np.abs(new_profits)
        return new_labels, new_weights

    def __init__(self, raw_data, margin, num_months, pricing_model: NormCallPricer, max_num_features=10,
                 model=None, feature_indexes=None, num_base_leaves=5, additional_feature_leaves=1, n_estimators=7):
        if model is None:
            model = clone(DEFAULT_MODEL_PROTOTYPE)
        self.data_provider = None
        self.raw_data = raw_data
        self.margin = margin
        self.num_months = num_months
        self.pricing_model = pricing_model
        # self.vol_factors = vol_factors
        # self.pricing_vol = pricing_vol
        self.feature_indexes = feature_indexes
        self.max_num_features = max_num_features
        self.model = model
        self.model_minus = clone(model)
        self.model_plus = clone(model)
        self.num_base_leaves = num_base_leaves
        self.additional_feature_leaves = additional_feature_leaves
        self.n_estimators = n_estimators
        self.prices = None
        self.final_values = None
        self.labels = None
        self.labels_plus = None
        self.labels_minus = None
        self.weights = None
        self.weights_plus = None
        self.weights_minus = None
        self.features_to_use = None
        self.transform = None
        self.training_history = []
        self.trained_advantage = None
        self.trained_value = None

    def assign_labels(self):
        # self.prices = 100 * self.raw_data.apply(calculate_prices, axis=1, margin=self.margin, num_months=self.num_months,
        #                                         vol_name=self.pricing_vol, vol_factor=self.vol_factors[0],
        #                                         base_factor=self.vol_factors[1])
        # growths = self.raw_data[GROWTH_NAMES[self.num_months]]
        growth_data = form_growth_df(self.raw_data, GROWTH_NAMES[self.num_months], self.pricing_model.vol_name)
        self.prices = 100 * self.pricing_model.find_expected_payouts_from_raw_margin(growth_data, self.margin)
        self.growths = 100 * growth_data['growth']
        # self.prices = self.pricing_model.calculate_prices(self.raw_data, threshold=self.margin)
        # self.final_values = calc_final_value(self.raw_data[GROWTH_NAMES[self.num_months]], threshold=self.margin)
        self.final_values = (self.growths > self.margin) * (self.growths - self.margin)
        profits = self.prices - self.final_values
        error = np.sqrt(np.mean(np.power(profits, 2)))
        data_block = self.raw_data.iloc[:, self.feature_indexes].copy()
        self.data_provider = ComboDataProvider(cont_names=list(data_block.columns), allow_cont_fill=False, cat_names=[],
                                               auxillary_names=[])
        self.labels = np.sign(profits)
        self.weights = np.abs(profits)
        print('Average profits were ', np.mean(profits))
        print('Rmse: ', error)
        data_block['weight'] = self.weights
        data_block['label'] = self.labels
        self.data_provider.ingest_data(data_block)

    def select_features(self, num_selection_bundles, results_evaluator: ResultEvaluator,
                        possible_indexes=None, established_indexes=None, **kwargs):
        if possible_indexes is None:
            possible_indexes = range(len(self.feature_indexes))
        if established_indexes is None:
            established_indexes = []
        end_selection = False
        while not end_selection and len(established_indexes) < self.max_num_features:
            print()
            round_number = len(established_indexes)
            print(round_number)
            bundle_providers = []
            for k in range(num_selection_bundles):
                splitter = TimeSeriesSplitter(forward_exclusion_length=7, backward_exclusion_length=10)
                my_bundle_provider = BasicBundleProvider(data_source=self.data_provider,
                                                         fixed_indexes=established_indexes,
                                                         splitter=splitter, **kwargs)
                my_bundle_provider.generate_trials()
                bundle_providers.append(my_bundle_provider)
            my_num_leaves = self.num_base_leaves + self.additional_feature_leaves * len(established_indexes)
            my_model = GradientBoostingClassifier(max_leaf_nodes=my_num_leaves, n_estimators=self.n_estimators)
            my_round = SingleFeatureEvaluationRound(data_provider=self.data_provider, model_prototype=my_model,
                                                    bundle_providers=bundle_providers, max_rows_better=3,
                                                    results_evaluator=results_evaluator.score, max_improvement=0.05,
                                                    established_indexes=established_indexes)
            my_round.compile_data()
            # my_round.summarize_results()
            my_summary = my_round.summary.sort_values('score', ascending=False)
            best_index, best_score, end_selection, candidates, summary = my_round.report_results()
            data_packet = (self, my_round, established_indexes)
            pickle.dump(data_packet, open('data_packet' + str(round_number) + '.pickle', 'wb'))
            self.training_history.append(my_summary)
            best_feature = self.data_provider.get_feature_name(int(best_index))
            print(best_index, best_feature, best_score, end_selection)
            print(datetime.now())
            this_candidates = list(candidates['idx'])
            if this_candidates:
                selection = this_candidates[0]
                if selection > -1:
                    established_indexes.append(selection)
        self.features_to_use = established_indexes
        last_round = self.training_history[-1].copy()
        last_round.sort_values('score', ascending=False, inplace=True)
        results = last_round.iloc[0]
        self.trained_advantage = results['advantage']
        self.trained_value = results['value']
        return self.training_history

    def create_data_set(self, set_transform=False, **kwargs):
        feature_data = self.raw_data.iloc[:, self.feature_indexes].copy()
        model_feature_data = feature_data.iloc[:, self.features_to_use].copy()
        t_features = model_feature_data.copy()
        if set_transform or self.transform is None:
            scaler = StandardScaler()
            t_features[:] = scaler.fit_transform(model_feature_data)
            if set_transform:
                self.transform = scaler.transform
        else:
            t_features[:] = self.transform(model_feature_data)
        if self.weights is None:
            my_fuzz = JitterSetGen(**kwargs)
            data_sets = my_fuzz.create_jitter_sets(t_features, dich_features=None, labels=self.labels, weights=None)
            #data_sets = create_jitters(features=t_features, labels=self.labels, **kwargs)
        else:
            my_fuzz = JitterSetGen(**kwargs)
            data_sets = my_fuzz.create_jitter_sets(t_features, dich_features=None, labels=self.labels,
                                                   weights=self.weights)
            #data_sets = create_jitters(features=t_features, labels=self.labels, weights=self.weights, **kwargs)
        full_data = pd.concat(data_sets, axis=0)
        full_data = full_data.dropna()
        return full_data

    def train(self, **kwargs):
        full_data = self.create_data_set(set_transform=True, **kwargs)
        if self.weights is None:
            full_features = full_data.iloc[:, : -1]
            full_labels = full_data.iloc[:, -1]
            minus_labels = full_labels - 0.5
            plus_labels = full_labels + 0.5
            self.model.fit(full_features, full_labels)
            self.model_plus.fit(full_features, plus_labels)
            self.model_minus.fit(full_features, minus_labels)
        else:
            full_features = full_data.iloc[:, : -2]
            full_weights = full_data.iloc[:, -2]
            full_labels = full_data.iloc[:, -1]
            plus_labels, plus_weights = self.translate_labels_and_weights(full_labels, full_weights, 0.5)
            minus_labels, minus_weights = self.translate_labels_and_weights(full_labels, full_weights, -0.5)
            self.model.fit(full_features, full_labels, sample_weight=full_weights)
            self.model_plus.fit(full_features, plus_labels, sample_weight = plus_weights)
            self.model_minus.fit(full_features, minus_labels, sample_weight = minus_weights)
        return full_data

    def predict_outcomes(self, data):
        # prices = 100 * data.apply(calculate_prices, axis=1, margin=self.margin, num_months=self.num_months,
        #                                     vol_name=self.pricing_vol, vol_factor=self.vol_factors[0],
        #                           base_factor=self.vol_factors[1])
        self.prices = self.pricing_model.calculate_prices(data, threshold=self.margin)
        feature_data = data.iloc[:, self.feature_indexes]
        model_feature_data = feature_data.iloc[:, self.features_to_use].copy()
        t_data = model_feature_data.copy()
        t_data[:] = self.transform(model_feature_data)
        results = self.model.predict_proba(t_data)[:, 1]
        plus_results = self.model_plus.predict_proba(t_data)[:, 1]
        minus_results = self.model_minus.predict_proba(t_data)[:, 1]
        return_df = pd.DataFrame({
            'outcome': results,
            'outcome_plus': plus_results,
            'outcome_minus': minus_results,
            'assumed_price': self.prices,
            'model_value': self.trained_value,
            'model_advantage': self.trained_advantage,
            'model_complexity': len(self.features_to_use)
        }, index=data.index)
        return return_df