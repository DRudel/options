from pricing import euro_vanilla
from features import prepare_data, GROWTH_DICT, GROWTH_NAMES, VOLATILITY_LIST, \
    calc_avg_abs_change, VOLATILITY_NAMES
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import sklearn.tree as tree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.base import clone
from itertools import combinations

my_rng = default_rng()

MIN_MARGIN_EQUIVALENT = 1.3
MAX_MARGIN_EQUIVALENT = 0.5
EVALUATION_EQUIVALENT = 0.75
# DEFAULT_MODEL_PROTOTYPE = tree.DecisionTreeRegressor(max_depth=5)
# DEFAULT_MODEL_PROTOTYPE = GradientBoostingRegressor(max_depth=3, init='zero', n_estimators=5)
DEFAULT_MODEL_PROTOTYPE = GradientBoostingClassifier(max_depth=3, init='zero', n_estimators=7)

def create_jitters(features, labels, jitter_count, jitter_magnitude, weights=None):
    labels = labels.copy()
    features = features.copy()
    base_data = features.copy()
    if weights is not None:
        base_data['weight'] = weights
    base_data['label'] = labels
    jitter_sets = [base_data]
    for k in range(jitter_count):
        this_random = 2 * jitter_magnitude * my_rng.random(features.shape)
        this_random = this_random - jitter_magnitude
        this_jitter = features + this_random
        if weights is not None:
            this_jitter['weight'] = weights
        this_jitter['label'] = labels
        jitter_sets.append(this_jitter)
    return jitter_sets
    # actual_data = features.copy()
    # actual_data['label'] = labels
    # jitters = [actual_data]
    # for k in range(jitter_count):
    #     this_random = 2 * jitter_magnitude * my_rng.random(features.shape)
    #     this_random = this_random - jitter_magnitude
    #     this_set = features + this_random
    #     this_set['label'] = labels
    #     jitters.append(this_set)
    # return jitters


def select_by_integer_index(df, selection, keep=True):
    idx = np.ones(len(df.index), dtype=bool)
    idx[selection] = False
    if keep:
        idx = ~idx
    return df.iloc[idx]

def calc_final_value(change, threshold):
    return_values = change.copy() - threshold
    return_values[return_values < 0] = 0
    return return_values


def calculate_prices(row, margin, vol_name, num_months, interest_rate=0.015, vol_factor=1):
    spot = row['price']
    strike = spot * (1 + margin/100)
    volatility = vol_factor * row[vol_name] / 100
    time = num_months / 12
    value = euro_vanilla(spot, strike, time, interest_rate, volatility)
    return value / spot


def evaluate_factor(factor, df, margin, num_months, vol_name, change_name, return_diff=False):
    clean_data = df.copy().dropna(subset=[vol_name, change_name])
    prices = 100 * clean_data.apply(calculate_prices, axis=1, margin=margin, num_months=num_months,
                              vol_name=vol_name, vol_factor=factor)
    values = calc_final_value(clean_data[change_name], margin)
    diff = prices - values
    squared_error = diff * diff
    if return_diff:
        return squared_error.mean(), diff.mean()
    return abs(diff.mean())


class Fund:

    def __init__(self, name:str, base_data: pd.DataFrame, feature_indexes=None):
        self.name = name
        self.data: pd.DataFrame = prepare_data(base_data)
        self.feature_indexes = feature_indexes
        self.features_to_use = dict()
        self.models = dict()
        self.average_volatility = calc_avg_abs_change(base_data['price'], 1).mean()
        self.eval_volatility_dict = dict()
        self.evaluation_margin_dict = dict()
        self.margin_dict = dict()
        self.vol_factor_dict = dict()
        for num_months in GROWTH_NAMES:
            tp_vol = np.sqrt(num_months) * self.average_volatility
            min_margin = int(np.floor(tp_vol / MIN_MARGIN_EQUIVALENT))
            max_margin = int(np.ceil(tp_vol / MAX_MARGIN_EQUIVALENT))
            self.margin_dict[num_months] = range(min_margin, max_margin + 1)
            self.evaluation_margin_dict[num_months] = np.round(tp_vol / EVALUATION_EQUIVALENT)

    def set_vol_factors(self, volatility_to_use=None):
        for num_months in GROWTH_NAMES:
            for margin in self.margin_dict[num_months]:
                self.set_vol_factor(margin, num_months, volatility_to_use=volatility_to_use)

    def set_vol_factor(self, margin, num_months, volatility_to_use=None):
        time_period = GROWTH_NAMES[num_months]
        if volatility_to_use is None:
            assert time_period in self.eval_volatility_dict
            volatility_to_use = self.eval_volatility_dict[time_period]
        solution = minimize(evaluate_factor, 5, args=(self.data, margin, num_months,
                                                      volatility_to_use, time_period),
                            options={'gtol': 1e-02}, tol=0.001)
        self.vol_factor_dict[(num_months, margin)] = solution.x[0]

    def set_evaluation_volatilities(self, volatilities_to_check=None):
        for num_months in GROWTH_NAMES:
            self.evaluate_volatility_periods(num_months, volatilities_to_check)

    def evaluate_volatility_periods(self, num_months: int, volatilities_to_check=None):
        assert num_months in list(GROWTH_NAMES.keys()), "time period not in growth dictionary"
        time_period = GROWTH_NAMES[num_months]
        if volatilities_to_check is None:
            volatilities_to_check = VOLATILITY_LIST
        best_error = None
        best_volatility = None
        for volatility_months in volatilities_to_check:
            vc = 'vol_' + str(volatility_months)
            errors = []
            print()
            print(vc)
            for margin in self.margin_dict[num_months]:
                solution = minimize(evaluate_factor, 5, args=(self.data, margin, num_months,
                                                              vc, time_period),
                                    options={'gtol': 1e-02}, tol=0.001)
                factor = solution.x
                err_data = evaluate_factor(factor, self.data, margin, num_months, vc, time_period,
                                           return_diff=True)
                errors.append(err_data[0])
            print(errors)
            mean_error = np.mean(errors)
            print(mean_error)
            if best_volatility is None or mean_error < best_error:
                best_error = mean_error
                best_volatility = vc
        self.eval_volatility_dict[time_period] = best_volatility


class TrainTestTrial:

    def __init__(self, data_sets: list[pd.DataFrame], exclusion_indexes, evaluation_indexes,
                 has_weights=False):
        training_sets = []
        evaluation_sets = []
        for ds in data_sets:
            training_data = select_by_integer_index(ds, exclusion_indexes, False)
            evaluation_data = select_by_integer_index(ds, evaluation_indexes)
            training_sets.append(training_data)
            evaluation_sets.append(evaluation_data)
        full_training_data = pd.concat(training_sets, axis=0)
        full_evaluation_data = pd.concat(evaluation_sets, axis=0)
        self.train_y = full_training_data.iloc[:, -1].copy()
        self.test_y = full_evaluation_data.iloc[:, -1].copy()
        if has_weights:
            self.train_X = full_training_data.iloc[:, : -2].copy()
            self.train_w = full_training_data.iloc[:, -2].copy()
            self.test_w = full_evaluation_data.iloc[:, -2].copy()
            self.test_X = full_evaluation_data.iloc[:, : -2].copy()
        else:
            self.train_X = full_training_data.iloc[:, : -1].copy()
            self.train_w = None
            self.test_w = None
            self.test_X = full_evaluation_data.iloc[:, : -1].copy()


class TrainTestBundle:

    def __init__(self, data, selection_size=None, exclusion_buffer_length=40,
                 jitter_count=0, jitter_magnitude=0.2, has_weights=False):
        self.data = data.dropna()
        self.labels = self.data.iloc[:, -1].copy()
        if selection_size is None:
            selection_size = int((len(data.index) - exclusion_buffer_length) / 10)
        self.selection_size = selection_size
        self.exclusion_buffer_length = exclusion_buffer_length
        self.jitter_count = jitter_count
        self.jitter_magnitude = jitter_magnitude
        self.trials: list[TrainTestTrial] = []
        self.has_weights = has_weights
        self.weights = None
        if self.has_weights:
            self.weights = self.data.iloc[:, -2].copy()

    def form_trials(self):
        if self.weights is None:
            temp_data = self.data.iloc[:, : -1].copy()
        else:
            temp_data = self.data.iloc[:, : -2].copy()
        scaler = StandardScaler()
        temp_data.loc[:, :] = scaler.fit_transform(temp_data)
        jitter_sets = create_jitters(temp_data, self.labels, self.jitter_count, self.jitter_magnitude, self.weights)
        #full_features, full_labels = create_jitters(temp_data, self.labels, self.jitter_count, self.jitter_magnitude)

        for fold_cutpoint in range(0, int(len(self.data.index) / self.selection_size)):
            cut_index = self.selection_size * fold_cutpoint
            before_cut = max(cut_index - self.exclusion_buffer_length, 0)
            exclusion_slice = slice(before_cut, cut_index + self.selection_size + self.exclusion_buffer_length)
            evaluation_slice = slice(cut_index, cut_index + self.selection_size)
            this_trial = TrainTestTrial(jitter_sets, exclusion_slice, evaluation_slice, has_weights=self.has_weights)
            self.trials.append(this_trial)


class FundModel:

    def __init__(self, data, margin, num_months, vol_factor, pricing_vol, model=None,
                 feature_indexes=None):
        if model is None:
            model = clone(DEFAULT_MODEL_PROTOTYPE)
        self.data = data
        self.margin = margin
        self.num_months = num_months
        self.vol_factor = vol_factor
        self.pricing_vol = pricing_vol
        self.feature_indexes = feature_indexes
        self.model = model
        self.prices = None
        self.final_values = None
        self.labels = None
        self.weights = None

    def assign_labels(self, classification=False):
        self.prices = 100 * self.data.apply(calculate_prices, axis=1, margin=self.margin, num_months=self.num_months,
                                            vol_name=VOLATILITY_NAMES[self.pricing_vol], vol_factor=self.vol_factor)

        self.final_values = calc_final_value(self.data[GROWTH_NAMES[self.num_months]],
                                             threshold=self.margin)
        profits = self.prices - self.final_values
        if classification:
            self.labels = np.sign(profits)
            self.weights = np.abs(profits)
        else:
            self.labels = profits

    def evaluate_features(self, data_sets: TrainTestBundle, selection_threshold=0, **kwargs):
        result_chunks = []
        for trial in data_sets.trials:
            this_result = self.run_trial(trial, **kwargs)
            result_chunks.append(this_result)
        full_results = pd.concat(result_chunks, axis=0)
        return full_results

    def run_trial(self, trial: TrainTestTrial, feature_indexes=None, **kwargs):
        if feature_indexes is None:
                assert self.feature_indexes is not None, "trainer does not have feature indexes set."
                feature_indexes = self.feature_indexes
        training_data = trial.train_X.copy()
        training_data = training_data.iloc[:, feature_indexes]
        has_weight = False
        if trial.train_w is not None:
            has_weight = True
            training_data['weight'] = trial.train_w.copy()
        training_data['label'] = trial.train_y.copy()
        self.train(training_data=training_data, has_weight=has_weight, **kwargs)
        evaluation_features = trial.test_X.iloc[:, feature_indexes]

        actuals = trial.test_y
        if trial.test_w is not None:
            predictions = self.model.predict_proba(evaluation_features)[:, 1]
            #predictions = 2 * predictions - 1
            actuals = actuals * trial.test_w
        else:
            predictions = self.model.predict(evaluation_features)
        results = pd.DataFrame(
            {
                'prediction': predictions,
                'actual': actuals
            }, index=trial.test_X.index
        )
        return results

    def train(self, training_data, has_weight=False, feature_indexes=None, use_all=True):
        if feature_indexes is None and use_all == False:
            assert self.feature_indexes is not None, "trainer does not have feature indexes set."
            feature_indexes = self.feature_indexes
        if not use_all:
            training_data = training_data.iloc[:, feature_indexes + [-1]].copy()
        training_data = training_data.dropna()
        training_labels = training_data.iloc[:, -1]
        if has_weight:
            training_features = training_data.iloc[:, : -2]
            training_weights = training_data.iloc[:, -2]
            self.model.fit(training_features, training_labels, sample_weight=training_weights)
        else:
            training_features = training_data.iloc[:, : -1]
            self.model.fit(training_features, training_labels)


class ResultEvaluator:

    def __init__(self, **kwargs):
        raise NotImplementedError

    def score(self, results_table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TanhResultEvaluator(ResultEvaluator):

    def __init__(self, factor, power, selection_threshold):
        self.factor = factor
        self.power = power
        self.threshold = selection_threshold

    def score(self, results_table: pd.DataFrame) -> pd.DataFrame:
        my_results = results_table.copy()
        my_results['trans_predictions'] = 2 * my_results['prediction'] - 1
        my_results['conviction'] = my_results['trans_predictions'].copy()
        my_results.loc[my_results['conviction'] < 0, 'conviction'] = 0
        my_results['conviction'] = np.tanh(self.factor * my_results['conviction'] ** self.power)
        my_results['profit_score'] = my_results['conviction'] * my_results['actual']
        result_count = len(my_results)
        selected_results = my_results.loc[my_results['trans_predictions'] > self.threshold, 'actual'].copy()
        play_count = len(selected_results) + 0.001
        play_rate = play_count / result_count
        loss = selected_results[selected_results < 0].sum()
        profit = selected_results.sum() + 0.001
        value = selected_results.mean()
        profit_score = my_results['profit_score'].sum()
        advantage = profit / (profit - loss)
        score = 0
        if profit > 0 and profit_score > 0 and advantage > 0:
            score = np.sqrt(profit_score * profit * advantage * advantage)
        test_index = list(my_results['idx'].unique())
        assert len(test_index) == 1, "multiple feature indexes"
        partner_index = list(my_results['partner'].unique())
        my_tuple = (test_index[0], partner_index[0])
        my_ml_index = pd.MultiIndex.from_tuples([my_tuple], names=['test_index', 'partner'])
        return_df = pd.DataFrame({
            'test_index': test_index,
            'partner_index': partner_index,
            'score':score,
            'advantage': advantage,
            'profit_score': profit_score,
            'profit': profit,
            'value': value,
            'play_rate': play_rate,
            'sample_size': result_count,
            'play_count': play_count,
            'loss': loss
        }, index=my_ml_index)
        return return_df


class FeatureEvaluationRound:

    def __init__(self, model_prototype, data, established_indexes, possible_indexes, num_bundles,
                 results_evaluator: ResultEvaluator, classification=True,
                 pairs_to_evaluate=None, test_singletons=True, **kwargs):
        self.model = clone(model_prototype)
        self.established_indexes = established_indexes
        self.possible_indexes = possible_indexes
        self.evaluator = results_evaluator
        self.trial_results = None
        self.classification = classification
        self.pairs_to_evaluate = pairs_to_evaluate
        self.test_singletons = test_singletons
        self.summary = None
        self.bundles: list[TrainTestBundle] = []
        for k in range(num_bundles):
            this_bundle = TrainTestBundle(data=data, **kwargs)
            this_bundle.form_trials()
            self.bundles.append(this_bundle)

    def get_test_pairs(self):
        remaining = set(range(len(self.possible_indexes))) - set(self.established_indexes)
        pairs_to_test = list(combinations(remaining, 2))
        if self.pairs_to_evaluate is not None:
            pairs_to_test = pairs_to_test[: self.pairs_to_evaluate]
        if self.test_singletons:
            pairs_to_test.extend([tuple([x, x]) for x in remaining])
        if len(self.established_indexes) > 0:
            dummy = self.established_indexes[0]
            pairs_to_test.append(tuple([dummy, dummy]))
        return pairs_to_test

    def compile_data(self):
        pairs_to_test = self.get_test_pairs()
        results_by_bundle = []
        for k, bundle in enumerate(self.bundles):
            bundle_results = self.traverse_pairs(pairs_to_test, bundle)
            bundle_results['bundle_index'] = k
            results_by_bundle.append(bundle_results)
        self.trial_results = pd.concat(results_by_bundle, axis=0)

    def traverse_pairs(self, pairs_to_test, bundle):
        index_cohorts = []
        for pair in pairs_to_test:
            feature_indexes = self.established_indexes + list(pair)
            this_cohort = self.process_bundle(bundle=bundle, feature_indexes=feature_indexes)
            index_cohorts.append(this_cohort)
        return pd.concat(index_cohorts, axis=0)

    def process_bundle(self, bundle, **kwargs):
        result_chunks = []
        for trial in bundle.trials:
            this_result = self.run_trial(trial, **kwargs)
            result_chunks.append(this_result)
        full_results = pd.concat(result_chunks, axis=0)
        return full_results

    def run_trial(self, trial: TrainTestTrial, feature_indexes, **kwargs):
        training_data = trial.train_X.copy()
        training_data = training_data.iloc[:, feature_indexes]
        has_weight = False
        if trial.train_w is not None:
            has_weight = True
            training_data['weight'] = trial.train_w.copy()
        training_data['label'] = trial.train_y.copy()
        self.train(training_data=training_data, has_weight=has_weight, **kwargs)
        evaluation_features = trial.test_X.iloc[:, feature_indexes]
        actuals = trial.test_y
        if self.classification:
            predictions = self.model.predict_proba(evaluation_features)[:, 1]
            if trial.test_w is not None:
                actuals = actuals * trial.test_w
        else:
            predictions = self.model.predict(evaluation_features)
        results = pd.DataFrame(
            {
                'first_index': feature_indexes[0],
                'second_index': feature_indexes[1],
                'prediction': predictions,
                'actual': actuals
            }, index=trial.test_X.index
        )
        return results

    def train(self, training_data, has_weight=False):
        training_data = training_data.dropna()
        training_labels = training_data.iloc[:, -1]
        if has_weight:
            training_features = training_data.iloc[:, : -2]
            training_weights = training_data.iloc[:, -2]
            self.model.fit(training_features, training_labels, sample_weight=training_weights)
        else:
            training_features = training_data.iloc[:, : -1]
            self.model.fit(training_features, training_labels)

    def summarize_results(self):
        results_by_index = []
        for k in self.possible_indexes:
            first_df = self.trial_results[self.trial_results['first_index'] == k].copy()
            first_df['partner'] = first_df['second_index'].copy()
            second_df = self.trial_results[self.trial_results['second_index'] == k].copy()
            second_df = second_df[second_df['first_index'] != second_df['second_index']]
            second_df['partner'] = second_df['first_index'].copy()
            indexed_df = pd.concat([first_df, second_df], axis=0)
            indexed_df['idx'] = k
            results_by_index.append(indexed_df)
        tagged_results = pd.concat(results_by_index, axis=0)
        summary = tagged_results.groupby(['idx','partner'], as_index=False,
                                         group_keys=False).apply(self.evaluator.score)
        self.summary = summary

