import pandas as pd
import numpy as np
from pricing import euro_vanilla
from features import GROWTH_NAMES
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import pickle
from utilities import calculate_prices, calc_final_value


DEFAULT_MODEL_PROTOTYPE = GradientBoostingClassifier(max_depth=3, init='zero', n_estimators=7)


class FundModel:

    @staticmethod
    def translate_labels_and_weights(labels, weights, translation):
        profits = labels * weights
        new_profits = profits + translation
        new_labels = np.sign(new_profits)
        new_weights = np.abs(new_profits)
        return new_labels, new_weights

    def __init__(self, data, margin, num_months, vol_factors, pricing_vol, model=None,
                 feature_indexes=None):
        if model is None:
            model = clone(DEFAULT_MODEL_PROTOTYPE)
        self.data = data
        self.margin = margin
        self.num_months = num_months
        self.vol_factors = vol_factors
        self.pricing_vol = pricing_vol
        self.feature_indexes = feature_indexes
        self.model = model
        self.model_minus = clone(model)
        self.model_plus = clone(model)
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

    def assign_labels(self, classification=True):
        self.prices = 100 * self.data.apply(calculate_prices, axis=1, margin=self.margin, num_months=self.num_months,
                                            vol_name=self.pricing_vol, vol_factor=self.vol_factors[0],
                                            base_factor=self.vol_factors[1])

        self.final_values = calc_final_value(self.data[GROWTH_NAMES[self.num_months]],
                                             threshold=self.margin)
        profits = self.prices - self.final_values
        if classification:
            self.labels = np.sign(profits)
            self.weights = np.abs(profits)
        else:
            self.labels = profits

    def select_features(self, possible_indexes=None, established_indexes=None, **kwargs):
        if possible_indexes is None:
            possible_indexes = range(len(self.feature_indexes))
        if established_indexes is None:
            established_indexes = []
        data_to_use = self.data.iloc[:, self.feature_indexes].copy()
        data_to_use['weight'] = self.weights
        data_to_use['label'] = self.labels
        k = 0
        best_index = -2
        end_selection = False # for early stopping because an isolated feature was the best addition
        while True:
            k += 1
            print(f'k = {k}')
            if k > 20:
                break
            if best_index > -1:
                established_indexes.append(best_index)
            if end_selection:
                break
            fer = FeatureEvaluationRound(self.model, data=data_to_use, established_indexes=established_indexes,
                                         possible_indexes=possible_indexes,
                                         backward_exclusion_length=self.num_months + 1,
                                         forward_exclusion_length=self.num_months + 1, **kwargs)
            fer.compile_data()
            fer.summarize_results()
            best_index, best_score, end_selection, summary = fer.report_results()
            self.training_history.append(summary)
            data_packet = (self, fer, established_indexes)
            pickle.dump(data_packet, open('data_packet' + str(k) + '.pickle', 'wb'))
        self.features_to_use = established_indexes
        last_round = self.training_history[-1].copy()
        last_round.sort_values('score', ascending=False, inplace=True)
        results = last_round.iloc[0]
        self.trained_advantage = results['advantage']
        self.trained_value = results['value']
        return self.training_history

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
        if self.weights is None:
            data_sets = create_jitters(features=t_features, labels=self.labels, **kwargs)
        else:
            data_sets = create_jitters(features=t_features, labels=self.labels, weights=self.weights, **kwargs)
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
        prices = 100 * data.apply(calculate_prices, axis=1, margin=self.margin, num_months=self.num_months,
                                            vol_name=self.pricing_vol, vol_factor=self.vol_factors[0],
                                  base_factor=self.vol_factors[1])
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
            'assumed_price': prices,
            'model_value': self.trained_value,
            'model_advantage': self.trained_advantage,
            'model_complexity': len(self.features_to_use)
        }, index=data.index)
        return return_df