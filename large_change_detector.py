# from features import prepare_data, NON_FEATURES, GROWTH_DICT
from features_v2 import GROWTH_DICT, NON_FEATURES, generate_full_data
import pandas as pd
import numpy as np
from ffs.data_provider import ComboDataProvider
from sklearn.ensemble import GradientBoostingClassifier
from result_evaluator import ResultEvaluator
from numpy.random import default_rng
from ffs.train_test import TimeIndexSplitter, BasicBundleProvider
from sklearn.base import clone
from ffs.feature_evaluation import SingleFeatureEvaluationRound
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from ffs.jitter import JitterSetGen
from sklearn.metrics import log_loss
import pickle

DEFAULT_MODEL_PROTOTYPE = GradientBoostingClassifier(n_estimators=7, random_state=173, max_depth=None)


def find_future_max_in_window(ser: pd.Series, timespan):
    reversed_data = ser[::-1].copy()
    max_values = reversed_data.rolling(timespan).max()
    return_series = max_values[::-1].copy()
    return return_series


class SimpleClassificationScorer:
    def score(self, results, idx):
        results.dropna(inplace=True)
        predictions = results['prediction']
        min_probs = results['mean_training_value'] / 5
        max_probs = 1 - (1 - results['mean_training_value']) / 5
        predictions = np.clip(predictions, min_probs, max_probs)
        actuals = results['actual']
        loss = log_loss(actuals, predictions)
        base_loss = log_loss(actuals, results['mean_training_value'])
        net_score = base_loss - loss
        return pd.DataFrame(
            {
            'idx': idx,
            'score': net_score,
            }, index=[idx]
        )


class LargeChangeDetector:

    def __init__(self, name: str, base_data: pd.DataFrame, pct_change_threshold: float, num_days: int,
                  feature_indexes=None, feature_prep=None, num_base_leaves=4, additional_feature_leaves=0.7,
                  max_num_features=10, decline=True, model_prototype=None):
        if model_prototype is None:
            model_prototype = clone(DEFAULT_MODEL_PROTOTYPE)
        if feature_prep is None:
            feature_prep = generate_full_data
        self.name = name
        self.full_data: pd.DataFrame = feature_prep(base_data)
        self._working_data = None
        self.model = clone(model_prototype)
        self.pct_change_threshold = pct_change_threshold
        self.num_days = num_days
        self.data_provider: ComboDataProvider = None
        self.feature_indexes = feature_indexes
        self.decline = decline
        self.feature_processor = feature_prep
        self.tuned_leaf_count = None
        self.num_base_leaves = num_base_leaves
        self.additional_feature_leaves = additional_feature_leaves
        self.max_num_features = max_num_features
        self.features_to_use = None
        self.transform = None
        self.labels = pd.Series(0, index=self.full_data.index)
        self.set_feature_indexes()

    def reset_working_data(self, frac):
        if frac is not None:
            self._working_data = self.full_data.sample(frac=frac, replace=False)
        else:
            self._working_data = self.full_data
        self._working_data.sort_index(inplace=True)
        self.set_data_provider()

    def set_feature_indexes(self):
        feature_indexes = list(range(len(self.full_data.columns)))
        for nf in NON_FEATURES:
            feature_indexes.remove(list(self.full_data.columns).index(nf))
        self.feature_indexes = feature_indexes

    def set_data_provider(self):
        # near_change = pd.DataFrame(index=self._working_data.index)
        max_date = self._working_data.index.max()
        max_valid_date = max_date - pd.Timedelta(str(self.num_days) + 'D')
        my_prices = self._working_data['close'].copy()
        if self.decline:
            my_prices[:] = -1 * my_prices[:]
        future_max = find_future_max_in_window(my_prices, str(self.num_days) + 'D')
        # if self.decline:
        #     future_max[:] = -1 * future_max[:]
        percent_change = 100 * ((my_prices - future_max) / my_prices).abs()
        # percent_change = 100 * (future_max / self._working_data['close'] - 1)
        # for (field, num_days_ahead) in GROWTH_DICT.items():
        #     if num_days_ahead <= self.num_days:
        #         near_change[field] = self._working_data[field]
        #     if self.decline:
        #         near_change[:] = -1 * near_change[:]
        # near_change['max_change'] = near_change.max(axis=1, skipna=True)
        self.labels = pd.Series(0, index=self._working_data.index)
        self.labels.loc[pd.isna(percent_change)] = None
        # near_change.dropna(subset=['max_change'], inplace=True)
        # self.labels.loc[near_change['max_change'] >= self.pct_change_threshold] = 1
        self.labels.loc[percent_change >= self.pct_change_threshold] = 1
        feature_data_block = self._working_data.iloc[:, self.feature_indexes].copy()
        data_block = feature_data_block.copy()
        data_block['label'] = self.labels
        data_block = data_block[data_block.index < max_valid_date].copy()
        self.data_provider = ComboDataProvider(cont_names=list(feature_data_block.columns), allow_cont_fill=False,
                                               cat_names=[], auxillary_names=[])
        self.data_provider.ingest_data(data_block, has_weights=False)
        print()

    def train(self, frac=None, **kwargs):
        self.reset_working_data(frac=frac)
        self.model.max_leaf_nodes = self.tuned_leaf_count
        training_data = self.create_data_set(set_transform=True, **kwargs)
        training_features = training_data.iloc[:, : -1]
        training_labels = training_data.iloc[:, -1]
        self.model.fit(training_features, training_labels)

    def select_leaf_count(self, results_evaluator: ResultEvaluator = None, frac=None,
                           features_to_use=None, min_leaves=2, allowed_fails=1, **kwargs):
        self.reset_working_data(frac=frac)
        if features_to_use is None:
            features_to_use = self.features_to_use
        if results_evaluator is None:
            results_evaluator = SimpleClassificationScorer()
        bundle_providers = self.get_bundles(data_provider=self.data_provider, fixed_indexes=features_to_use, **kwargs)
        best_score = None
        best_leaf_count = None
        best_summary = None
        num_leaves = min_leaves
        num_fails = -1
        while num_fails < allowed_fails:
            my_model = clone(self.model)
            my_model.max_leaf_nodes = num_leaves
            my_round = SingleFeatureEvaluationRound(data_provider=self.data_provider, model_prototype=my_model,
                                                    bundle_providers=bundle_providers, max_indexes_better=3,
                                                    results_evaluator=results_evaluator.score, max_improvement=0.005,
                                                    established_indexes=features_to_use,
                                                    use_probability=True)
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
        feature_data = self._working_data.iloc[:, self.feature_indexes].copy()
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

    def select_features(self, results_evaluator: ResultEvaluator = None, established_indexes=None, min_features=4,
                        frac=None, **kwargs):
        self.reset_working_data(frac=frac)
        if results_evaluator is None:
            results_evaluator = SimpleClassificationScorer()
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
                                                    results_evaluator=results_evaluator.score, max_improvement=0.005,
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
                    forward_exclusion_timedelta=np.timedelta64(210, 'D'), max_offset=np.timedelta64(1000, 'D'),
                    backward_exclusion_timedelta=np.timedelta64(300, 'D'), num_trials=None, **kwargs):
        if data_provider is None:
            data_provider = self.data_provider
        my_master_rng = None
        if master_seed is not None:
            my_master_rng = default_rng(master_seed)
        bundle_providers = []
        for k in range(num_selection_bundles):
            splitter = TimeIndexSplitter(forward_exclusion_timedelta=forward_exclusion_timedelta,
                                         backward_exclusion_timedelta=backward_exclusion_timedelta,
                                         num_trials=num_trials, max_offset=max_offset)
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
        model_features = features.iloc[:, self.features_to_use].copy()
        model_features.dropna(inplace=True)
        t_data = model_features.copy()
        t_data[:] = self.transform(model_features)
        probs = self.model.predict_proba(t_data)
        model_features['date'] = t_data.index.copy()
        model_features['danger'] = probs[:, 1].copy()
        return model_features

    def save(self, memo: str = None):
        if memo is None:
            memo = str(datetime.now())[:19]
        memo = memo.replace(':', '_')
        pickle.dump(self, open('lc_detectors/' + self.name + '_' + memo + '.pickle', 'wb'))