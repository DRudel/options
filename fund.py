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
from sklearn.base import clone

my_rng = default_rng()

MIN_MARGIN_EQUIVALENT = 1.3
MAX_MARGIN_EQUIVALENT = 0.5
EVALUATION_EQUIVALENT = 0.75
DEFAULT_MODEL_PROTOTYPE = tree.DecisionTreeRegressor(max_depth=5)


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

    def __init__(self, data_sets: list[pd.DataFrame], exclusion_indexes, evaluation_indexes):
        training_sets = []
        evaluation_sets = []
        for ds in data_sets:
            training_data = select_by_integer_index(ds, exclusion_indexes, False)
            evaluation_data = select_by_integer_index(ds, evaluation_indexes)
            training_sets.append(training_data)
            evaluation_sets.append(evaluation_data)
        full_training_data = pd.concat(training_sets, axis=0)
        full_evaluation_data = pd.concat(evaluation_sets, axis=0)
        self.train_X = full_training_data.iloc[:, : -1].copy()
        self.train_y = full_training_data.iloc[:, -1].copy()
        self.test_X = full_evaluation_data.iloc[:, : -1].copy()
        self.test_y = full_evaluation_data.iloc[:, -1].copy()


class TrainTestBundle:

    def __init__(self, data, selection_size=None, exclusion_buffer_length=40,
                 jitter_count=0, jitter_magnitude=0.2):
        self.data = data.dropna()
        self.labels = self.data.iloc[:, -1].copy()
        if selection_size is None:
            selection_size = int((len(data.index) - exclusion_buffer_length) / 10)
        self.selection_size = selection_size
        self.exclusion_buffer_length = 40
        self.jitter_count = jitter_count
        self.jitter_magnitude = jitter_magnitude
        self.trials: list[TrainTestTrial] = []

    def form_trials(self):
        temp_data = self.data.iloc[:, : -1].copy()
        scaler = StandardScaler()
        temp_data.loc[:, :] = scaler.fit_transform(temp_data)
        base_data = temp_data.copy()
        base_data['label'] = self.labels
        jitter_sets = [base_data]
        for k in range(self.jitter_count):
            this_random = 2 * self.jitter_magnitude * my_rng.random(temp_data.shape)
            this_random = this_random - self.jitter_magnitude
            this_jitter = temp_data + this_random
            this_jitter['label'] = self.labels
            jitter_sets.append(this_jitter)

        for fold_cutpoint in range(0, int(len(self.data.index) / self.selection_size)):
            cut_index = 40 * fold_cutpoint
            before_cut = max(cut_index - self.exclusion_buffer_length, 0)
            exclusion_slice = slice(before_cut, cut_index + self.selection_size + self.exclusion_buffer_length)
            evaluation_slice = slice(cut_index, cut_index + self.selection_size)
            this_trial = TrainTestTrial(jitter_sets, exclusion_slice, evaluation_slice)
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

    def assign_labels(self):
        self.prices = 100 * self.data.apply(calculate_prices, axis=1, margin=self.margin,
                                       num_months=self.num_months,
                                       vol_name=VOLATILITY_NAMES[self.pricing_vol],
                                       vol_factor=self.vol_factor)

        self.final_values = calc_final_value(self.data[GROWTH_NAMES[self.num_months]],
                                             threshold=self.margin)
        self.labels = self.prices - self.final_values

    def evaluate_model(self, exclusion_buffer_length=40, selection_size=40, **kwargs):
        result_chunks = []
        for fold_cutpoint in range(0, int(len(self.data.index) / selection_size)):
            cut_index = 40 * fold_cutpoint
            before_cut = max(cut_index - exclusion_buffer_length, 0)
            exclusion_slice = slice(before_cut, cut_index + selection_size + exclusion_buffer_length)
            evaluation_slice = slice(cut_index, cut_index + selection_size)
            this_selected_labels = self.evaluate_slice(exclusion_indexes=exclusion_slice,
                                                       evaluation_indexes=evaluation_slice, **kwargs)
            result_chunks.append(this_selected_labels)
        full_results = pd.concat(result_chunks, axis=0)
        return full_results

    def evaluate_slice(self, exclusion_indexes, evaluation_indexes, feature_indexes=None, **kwargs):
        assert self.labels is not None, "must assign labels before training"
        if feature_indexes is None:
            assert self.feature_indexes is not None, "trainer does not have feature indexes set."
            feature_indexes = self.feature_indexes
        temp_data = self.data.copy()
        temp_data = temp_data.iloc[:, feature_indexes].copy()
        scaler = StandardScaler()
        temp_data.loc[:, :] = scaler.fit_transform(temp_data)
        temp_data['labels'] = self.labels
        temp_data = temp_data.dropna()
        training_data = select_by_integer_index(temp_data, exclusion_indexes, False)
        evaluation_data = select_by_integer_index(temp_data, evaluation_indexes)
        evaluation_features = evaluation_data.iloc[:, : -1].copy()
        evaluation_labels = evaluation_data.iloc[:, -1].copy()
        self.train(training_data, **kwargs)
        predictions = self.model.predict(evaluation_features)
        results = pd.DataFrame(
            {
                'prediction': predictions,
                'actual': evaluation_labels
            }, index=evaluation_data.index
        )
        return results

    def train(self, training_data, feature_indexes=None, use_all=True,
              jitter_count=0, jitter_magnitude=0.2):
        if feature_indexes is None and use_all == False:
            assert self.feature_indexes is not None, "trainer does not have feature indexes set."
            feature_indexes = self.feature_indexes
        if not use_all:
            training_data = training_data.iloc[:, feature_indexes + [-1]].copy()
        training_data = training_data.dropna()
        training_features = training_data.iloc[:, :-1]
        training_labels = training_data.iloc[:, -1]
        if jitter_count > 0:
            jitters = [training_features]
            bag_of_labels = [training_labels]
            for k in range(jitter_count):
                this_random = 2 * jitter_magnitude * my_rng.random(training_features.shape)
                this_random = this_random - jitter_magnitude
                jitters.append(training_features + this_random)
                bag_of_labels.append(training_labels)
            training_features = np.vstack(jitters)
            training_labels = np.hstack(bag_of_labels)
        self.model.fit(training_features, training_labels)
