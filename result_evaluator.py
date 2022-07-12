import pandas as pd
import numpy as np


class ResultEvaluator:

    def __init__(self, **kwargs):
        raise NotImplementedError

    def score(self, results_table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SingleTanhResultEvaluator(ResultEvaluator):

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
        return_df = pd.DataFrame({
            'idx': test_index.copy(),
            'score':score,
            'advantage': advantage,
            'profit_score': profit_score,
            'profit': profit,
            'value': value,
            'play_rate': play_rate,
            'sample_size': result_count,
            'play_count': play_count,
            'loss': loss
        }, index=test_index)
        return return_df

class DoubleTanhResultEvaluator(ResultEvaluator):

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
        my_ml_index = pd.MultiIndex.from_tuples([my_tuple], names=['t_idx', 'p_idx'])
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