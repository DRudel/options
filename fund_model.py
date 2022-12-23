import pandas as pd
import numpy as np
from pricing import euro_vanilla
from features import GROWTH_NAMES
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import lightgbm as lgbm
import pickle
from numpy.random import default_rng
from utilities import calc_final_value, form_pricing_data
from ffs.feature_evaluation import SingleFeatureEvaluationRound
from ffs.data_provider import ComboDataProvider
from ffs.train_test import TimeSeriesSplitter, BasicBundleProvider
from ffs.jitter import JitterSetGen
from result_evaluator import ResultEvaluator
from datetime import datetime
from pricing_models import NormPricer

DEFAULT_CLASSIFICATION_PROTOTYPE = GradientBoostingClassifier(n_estimators=7, random_state=173, max_depth=None)
DEFAULT_REGRESSION_PROTOTYPE = GradientBoostingRegressor(n_estimators=7, random_state=113, max_depth=None)


class FundModel:

    @staticmethod
    def translate_labels_and_weights(labels, weights, prices, factor):
        '''
        Calculates new labels and weights assuming a change in prices.
        :param labels: binary profit or loss
        :param weights: original weights (= magnitude of profit/loss)
        :param prices: original prices
        :param factor: change in prices
        :return: new labels and weights
        '''
        profits = labels * weights
        new_profits = profits + (factor - 1) * prices
        new_labels = np.sign(new_profits)
        new_weights = np.abs(new_profits)
        return new_labels, new_weights

    def __init__(self, raw_data, margin, num_months, pricing_model: NormPricer, max_num_features=10,
                 classification_model=None, regression_model=None,
                 feature_indexes=None, num_base_leaves=3, additional_feature_leaves=1.7,
                 num_selection_estimators=7, dear_price_modifier=1.4, cheap_price_modifier=0.7):
        if classification_model is None:
            classification_model = clone(DEFAULT_CLASSIFICATION_PROTOTYPE)
        if regression_model is None:
            regression_model = clone(DEFAULT_REGRESSION_PROTOTYPE)
        self.classification_data_provider = None
        self.regression_data_provider = None
        self.raw_data = raw_data
        self.margin = margin
        self.num_months = num_months
        self.pricing_model = pricing_model
        self.feature_indexes = feature_indexes
        self.max_num_features = max_num_features
        self.dear_price_modifier = dear_price_modifier
        self.cheap_price_modifier = cheap_price_modifier
        self.classification_model = classification_model
        self.num_leaves_classification = None
        self.classification_model_cheap = clone(classification_model)
        self.classification_model_dear = clone(classification_model)
        self.regression_model = regression_model
        self.num_leaves_regression = None
        self.num_base_leaves = num_base_leaves
        self.additional_feature_leaves = additional_feature_leaves
        self.num_selection_estimators = num_selection_estimators
        self.prices = None
        self.final_values = None
        self.labels = None
        self.weights = None
        self.features_to_use = None
        self.transform = None
        self.training_history = []
        self.trained_advantage = None
        self.trained_value = None
        self.price_idx = None
        self.offset = None

    def assign_labels(self):
        growth_data = form_pricing_data(self.raw_data, GROWTH_NAMES[self.num_months], self.pricing_model.vol_name)
        self.prices = 100 * self.pricing_model.find_expected_payouts_from_raw_margin(growth_data, self.margin)
        self.growths = 100 * growth_data['growth']
        self.final_values = (self.growths > self.margin) * (self.growths - self.margin)
        profits = self.prices - self.final_values
        self.pricing_offset = np.mean(profits)
        profits = profits - self.pricing_offset
        error = np.sqrt(np.mean(np.power(profits, 2)))
        data_block = self.raw_data.iloc[:, self.feature_indexes].copy()
        classification_data_block = data_block.copy()
        self.labels = np.sign(profits)
        self.weights = np.abs(profits)
        classification_data_block['weight'] = self.weights
        classification_data_block['label'] = self.labels
        self.classification_data_provider = ComboDataProvider(cont_names=list(data_block.columns),
                                                               allow_cont_fill=False, cat_names=[], auxillary_names=[])
        self.classification_data_provider.ingest_data(classification_data_block)
        print('Average profits were ', np.mean(profits))
        print('Rmse: ', error)
        regression_data_block = data_block.copy()
        # regression_data_block['assumed_price'] = self.prices
        regression_data_block['label'] = profits
        self.regression_data_provider = ComboDataProvider(cont_names=list(data_block.columns),
                                                          allow_cont_fill=False, cat_names=[], auxillary_names=[])
        self.regression_data_provider.ingest_data(regression_data_block, has_weights=False)
        # self.price_idx = self.regression_data_provider.get_feature_index('assumed_price')
        return_df = pd.DataFrame({
            'growth': self.growths,
            'price': self.prices,
            'offset': self.pricing_offset,
            'value': self.final_values,
            'profit': profits
        })
        return return_df


    def select_features(self, results_evaluator: ResultEvaluator, possible_indexes=None, established_indexes=None,
                        min_features=4, **kwargs):
        if possible_indexes is None:
            possible_indexes = range(len(self.feature_indexes))
        if established_indexes is None:
            established_indexes = []
        end_selection = False
        previous_score = None
        while not end_selection and len(established_indexes) < self.max_num_features:
            # my_master_rng = None
            # if master_seed is not None:
            #     my_master_rng = default_rng(master_seed)
            # print()
            round_number = len(established_indexes)
            print(round_number)
            bundle_providers = self.get_bundles(fixed_indexes=established_indexes, **kwargs)
            # bundle_providers = []
            # for k in range(num_selection_bundles):
            #     splitter = TimeSeriesSplitter(forward_exclusion_length=7, backward_exclusion_length=10)
            #     my_bundle_provider = BasicBundleProvider(data_source=self.data_provider,
            #                                              fixed_indexes=established_indexes,
            #                                              splitter=splitter, rng=my_master_rng, **kwargs)
            #     trial_random_state = None
            #     if my_master_rng is not None:
            #         trial_random_state = int(1000 * my_master_rng.random())
            #     my_bundle_provider.generate_trials(random_state=trial_random_state)
            #     bundle_providers.append(my_bundle_provider)
            num_leaves = int(self.num_base_leaves + self.additional_feature_leaves * len(established_indexes))
            print(f'using {num_leaves} leaves. Previous score is {previous_score}')
            my_model = clone(self.classification_model)
            my_model.max_leaf_nodes = num_leaves
            # my_model = GradientBoostingClassifier(max_leaf_nodes=my_num_leaves, n_estimators=self.n_estimators,
            #                                       max_depth=None)
            my_round = SingleFeatureEvaluationRound(data_provider=self.classification_data_provider,
                                                    model_prototype=my_model,
                                                    bundle_providers=bundle_providers, max_indexes_better=3,
                                                    results_evaluator=results_evaluator.score, max_improvement=0.05,
                                                    established_indexes=established_indexes, min_features=min_features)
            my_round.compile_data()
            my_summary = my_round.summary.sort_values('score', ascending=False)
            best_index, best_score, end_selection, candidates, summary = my_round.report_results()
            data_packet = (self, my_round, established_indexes)
            # pickle.dump(data_packet, open('data_packet' + str(round_number) + '.pickle', 'wb'))
            #self.training_history.append(my_summary)
            best_feature = self.classification_data_provider.get_feature_name(int(best_index))
            print(best_index, best_feature, best_score, end_selection)
            if previous_score is not None:
                if best_score > previous_score:
                    previous_score = best_score
                elif len(established_indexes) >= min_features:
                    print('previous score better than current. Aborting')
                    end_selection = True
                    continue
            else:
                previous_score = best_score
            print(datetime.now())
            this_candidates = list(candidates['idx'])
            if this_candidates:
                selection = this_candidates[0]
                if selection > -1:
                    established_indexes.append(selection)
        if len(established_indexes) == 0:
            print("no indexes selected")
        self.features_to_use = established_indexes
        #return self.training_history

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
        aux_data = pd.DataFrame({
            'price': self.prices,
            'weight': self.weights,
            'label': self.labels
        }, index=t_features.index)
        my_fuzz = JitterSetGen(**kwargs)
        data_sets = my_fuzz.create_jitter_sets(t_features, dich_features=None, aux_data=aux_data)
        full_data = pd.concat(data_sets, axis=0)
        full_data = full_data.dropna()
        return full_data

    def train_classifier(self, **kwargs):
        self.classification_model.max_leaf_nodes = self.num_leaves_classification
        self.classification_model_cheap = clone(self.classification_model)
        self.classification_model_dear = clone(self.classification_model)
        full_data = self.create_data_set(set_transform=True, **kwargs)
        full_features = full_data.iloc[:, : -3]
        full_prices = full_data.iloc[:, -3]
        full_weights = full_data.iloc[:, -2]
        full_labels = full_data.iloc[:, -1]
        dear_labels, dear_weights = self.translate_labels_and_weights(full_labels, full_weights, full_prices,
                                                                      self.dear_price_modifier)
        middle_labels, middle_weights = self.translate_labels_and_weights(full_labels, full_weights, full_prices,
                                                                      1)
        cheap_labels, cheap_weights = self.translate_labels_and_weights(full_labels, full_weights, full_prices,
                                                                        self.cheap_price_modifier)
        self.classification_model.fit(full_features, middle_labels, sample_weight=middle_weights)
        self.classification_model_dear.fit(full_features, dear_labels, sample_weight=dear_weights)
        self.classification_model_cheap.fit(full_features, cheap_labels, sample_weight=cheap_weights)
        # cheap_results = self.classification_model_cheap.predict_proba(full_features)
        # middle_results = self.classification_model.predict_proba(full_features)
        # dear_results = self.classification_model_dear.predict_proba(full_features)
        return full_data

    def train_regressor(self, **kwargs):
        data = self.create_data_set(set_transform=True, **kwargs)
        features = data.iloc[:, : -3]
        prices = data.iloc[:, -3]
        weights = data.iloc[:, -2]
        labels = data.iloc[:, -1]
        profits = weights * labels
        self.regression_model.max_leaf_nodes = self.num_leaves_regression
        self.regression_model.fit(features, profits)

    def predict_outcomes(self, data, num_days_offset=0):
        pricing_data = form_pricing_data(data, GROWTH_NAMES[self.num_months], self.pricing_model.vol_name,
                                         include_growths=False)
        pricing_data['time'] = pricing_data['time'] + num_days_offset
        num_days = pricing_data['time'].iloc[0]
        prices = 100 * self.pricing_model.find_expected_payouts_from_raw_margin(pricing_data, self.margin)
        prices = prices - self.pricing_offset
        # self.prices = self.pricing_model.calculate_prices(data, threshold=self.margin)
        feature_data = data.iloc[:, self.feature_indexes]
        model_feature_data = feature_data.iloc[:, self.features_to_use].copy()
        t_data = model_feature_data.copy()
        t_data[:] = self.transform(model_feature_data)
        classification_results = self.classification_model.predict_proba(t_data)[:, 1]
        classification_dear_results = self.classification_model_dear.predict_proba(t_data)[:, 1]
        classification_cheap_results = self.classification_model_cheap.predict_proba(t_data)[:, 1]
        regression_results = self.regression_model.predict(t_data)
        return_df = pd.DataFrame({
            'num_days': num_days,
            'prob': classification_results,
            'prob_dear': classification_dear_results,
            'prob_cheap': classification_cheap_results,
            'estimated_profit': regression_results,
            'assumed_price': prices,
            'pricing_offset': self.pricing_offset,
            'model_value': self.trained_value,
            'model_advantage': self.trained_advantage,
            'model_complexity': len(self.features_to_use)
        }, index=data.index)
        return return_df

    def select_leaf_count(self, model, data_provider, classification, results_evaluator: ResultEvaluator,
                           features_to_use=None, min_leaves=None, allowed_fails=1, **kwargs):
        if min_leaves is None:
            min_leaves = len(self.features_to_use)
        if min_leaves < 2:
            min_leaves = 2
        if features_to_use is None:
            features_to_use = self.features_to_use
        bundle_providers = self.get_bundles(data_provider=data_provider, fixed_indexes=features_to_use, **kwargs)
        best_score = None
        best_leaf_count = None
        best_summary = None
        num_leaves = min_leaves
        num_fails = -1
        while num_fails < allowed_fails:
            my_model = clone(model)
            my_model.max_leaf_nodes = num_leaves
            my_round = SingleFeatureEvaluationRound(data_provider=data_provider, model_prototype=my_model,
                                                    bundle_providers=bundle_providers, max_indexes_better=3,
                                                    results_evaluator=results_evaluator.score, max_improvement=0.05,
                                                    established_indexes=features_to_use,
                                                    use_probability=classification)
            my_round.compile_data(no_new_features=True)
            this_score = my_round.summary['score'].iloc[0]
            if best_score is None or this_score > best_score:
                best_score = this_score
                best_leaf_count = num_leaves
                best_summary = my_round.summary
                num_fails = -1
            else:
                num_fails += 1
            print(f'num_features = {len(self.features_to_use)}, num_leaves = {num_leaves}; score = {this_score}; '
                  f'num_fails = {num_fails}', datetime.now())
            num_leaves += 1
        # self.num_leaves = best_leaf_count
        # self.trained_advantage = best_summary['advantage'].iloc[0]
        # self.trained_value = best_summary['value'].iloc[0]
        return best_leaf_count, best_summary

    def get_bundles(self, num_selection_bundles, fixed_indexes, data_provider=None, master_seed=None,
                    forward_exclusion_length=7, backward_exclusion_length=10, **kwargs):
        if data_provider is None:
            data_provider = self.classification_data_provider
        my_master_rng = None
        if master_seed is not None:
            my_master_rng = default_rng(master_seed)
        bundle_providers = []
        for k in range(num_selection_bundles):
            splitter = TimeSeriesSplitter(forward_exclusion_length=forward_exclusion_length,
                                          backward_exclusion_length=backward_exclusion_length)
            my_bundle_provider = BasicBundleProvider(data_source=data_provider,
                                                     fixed_indexes=fixed_indexes,
                                                     splitter=splitter, rng=my_master_rng, **kwargs)
            trial_random_state = None
            if my_master_rng is not None:
                trial_random_state = int(1000 * my_master_rng.random())
            my_bundle_provider.generate_trials(random_state=trial_random_state)
            bundle_providers.append(my_bundle_provider)
        return bundle_providers