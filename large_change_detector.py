from features import prepare_data, NON_FEATURES, GROWTH_DICT
import pandas as pd
import numpy as np
from ffs.data_provider import ComboDataProvider
from sklearn.ensemble import GradientBoostingClassifier
from result_evaluator import ResultEvaluator
from numpy.random import default_rng
from ffs.train_test import TimeSeriesSplitter, BasicBundleProvider
from sklearn.base import clone
from ffs.feature_evaluation import SingleFeatureEvaluationRound
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from ffs.jitter import JitterSetGen

DEFAULT_MODEL_PROTOTYPE = GradientBoostingClassifier(n_estimators=7, random_state=173, max_depth=None)


class LargeChangeDetector:

    def __iniit__(self, name: str, base_data: pd.DataFrame, pct_change_threshold: float, num_months: int,
                  feature_indexes=None, feature_prep=None, num_base_leaves=3, additional_feature_leaves=1.7,
                  max_num_features=10, decline=True, model_prototype=None):
        if model_prototype is None:
            model_prototype = clone(DEFAULT_MODEL_PROTOTYPE)
        if feature_prep is None:
            feature_prep = prepare_data
        self.name = name
        self.data: pd.DataFrame = feature_prep(base_data)
        self.model = clone(model_prototype)
        self.pct_change_threshold = pct_change_threshold
        self.num_months = num_months
        self.classification_data_provider: ComboDataProvider
        self.feature_indexes = feature_indexes
        self.decline = decline
        self.feature_processor = feature_prep
        self.tuned_leaf_count = None
        self.num_base_leaves = num_base_leaves
        self.additional_feature_leaves = additional_feature_leaves
        self.max_num_features = max_num_features
        self.features_to_use = None
        self.transform = None
        self.labels = pd.Series(0, index=self.data)
        self.set_feature_indexes()

    def set_feature_indexes(self):
        feature_indexes = list(range(len(self.data.columns)))
        for nf in NON_FEATURES:
            feature_indexes.remove(list(self.data.columns).index(nf))
        self.feature_indexes = feature_indexes

    def assign_labels(self):
        near_change = pd.DataFrame(index=self.data.index)
        for (field, num_months_ahead) in GROWTH_DICT.items():
            if num_months_ahead <= self.num_months:
                near_change[field] = self.data[field]
            if self.decline:
                near_change[:] = -1 * near_change[:]
        near_change['max_change'] = near_change.max(axis=1)
        self.labels.loc[near_change['max_change'] >= self.pct_change_threshold] = 1
        data_block = self.data.iloc[:, self.feature_indexes].copy()
        data_block['label'] = self.labels
        self.data_provider = ComboDataProvider(cont_names=list(data_block.columns), allow_cont_fill=False,
                                               cat_names=[], auxillary_names=[])

    def train(self, **kwargs):
        self.model.max_leaf_nodes = self.tuned_leaf_count
        training_data = self.create_data_set(set_transform=True, **kwargs)
        training_features = training_data.iloc[:, : -1]
        training_labels = training_data.iloc[:, -1]
        self.model.fit(training_features, training_labels)

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
        self.tuned_leaf_count = best_leaf_count

    def create_data_set(self, set_transform=False, **kwargs):
        feature_data = self.data.iloc[:, self.feature_indexes].copy()
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
            'label': self.labels
        }, index=t_features.index)
        my_fuzz = JitterSetGen(**kwargs)
        data_sets = my_fuzz.create_jitter_sets(t_features, dich_features=None, aux_data=aux_data)
        full_data = pd.concat(data_sets, axis=0)
        full_data = full_data.dropna()
        return full_data

    def select_features(self, results_evaluator: ResultEvaluator, established_indexes=None, min_features=4, **kwargs):
        if established_indexes is None:
            established_indexes = []
        end_selection = False
        previous_score = None
        while not end_selection and len(established_indexes) < self.max_num_features:
            round_number = len(established_indexes)
            print(round_number)
            bundle_providers = self.get_bundles(fixed_indexes=established_indexes, **kwargs)
            num_leaves = int(self.num_base_leaves + self.additional_feature_leaves * len(established_indexes))
            print(f'using {num_leaves} leaves. Previous score is {previous_score}')
            my_model = clone(self.model)
            my_model.max_leaf_nodes = num_leaves
            my_round = SingleFeatureEvaluationRound(data_provider=self.data_provider,
                                                    model_prototype=my_model,
                                                    bundle_providers=bundle_providers, max_indexes_better=3,
                                                    results_evaluator=results_evaluator.score, max_improvement=0.05,
                                                    established_indexes=established_indexes, min_features=min_features)
            my_round.compile_data()
            my_summary = my_round.summary.sort_values('score', ascending=False)
            best_index, best_score, end_selection, candidates, summary = my_round.report_results()
            best_feature = self.data_provider.get_feature_name(int(best_index))
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

    def get_bundles(self, num_selection_bundles, fixed_indexes, data_provider=None, master_seed=None,
                    forward_exclusion_length=7, backward_exclusion_length=10, **kwargs):
        if data_provider is None:
            data_provider = self.data_provider
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

    def predict(self, data):
        processed_data = self.feature_processor(data)
        features = processed_data.iloc[:, self.feature_indexes].copy()
        probs = self.model.predict_proba(features)
        features['data'] = data.iloc[:, 0]
        features['danger'] = probs[:, 1]
        return features