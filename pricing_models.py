from scipy.stats import norm, percentileofscore
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from functools import partial
from time import time
from numpy.random import default_rng

my_rng = default_rng()


class NormCallPricer:

    def __init__(self, param_scale=10, reg_vol_var_sigma=0, reg_base_mu=0, reg_excluded_prop=0):
        self.param_scale = param_scale
        self.base_sigma = 0.006
        self.vol_var_sigma = 0
        self.base_mu = 0.0
        self.excluded_prop = 0.01
        self.n_iter = 0
        self.reg_vol_var_sigma = reg_vol_var_sigma
        self.reg_base_mu = reg_base_mu
        self.reg_excluded_prop = reg_excluded_prop

    def set_params(self, params):
        for n, field in enumerate(['base_sigma', 'excluded_prop', 'vol_var_sigma', 'base_mu']):
            if n < len(params):
                self.__setattr__(field, params[n] / self.param_scale)
        if self.base_mu < 0:
            self.base_mu = 0
        if self.base_sigma < 0:
            self.base_sigma = 0
        if self.vol_var_sigma < 0:
            self.vol_var_sigma = 0
        if self.excluded_prop < 0:
            self.excluded_prop = 0 * self.excluded_prop

    def get_shape_params(self, time=1, vol=0):
        '''
        get_shape_params creates a _miniaturized_ probability distribution equal to 1/time * distribtuion we actually
        care about. We care about the distribution (daily_mu) * time with sigma = daily_sigma * sqrt(time), but to
        normalize all data to a daily distribution we divide by time to get a distribution with mu = daily mu and
        sigma = daily_sigma / sqrt(time)
        :param time:
        :param vol:
        :return:
        '''
        sigma = (self.base_sigma + self.vol_var_sigma * vol) / np.sqrt(time)
        sigma = np.clip(sigma, 0.0001, 1)
        mu = self.base_mu
        return mu, sigma

    def create_distribution(self, ser: pd.Series):
        mu, sigma = self.get_shape_params(ser['time'], ser['vol'])
        x = np.linspace(-0.005, 0.005, 800)
        norm_pdf = norm.pdf(x, mu, sigma)
        probabilities = (1 - self.excluded_prop) * norm_pdf / np.sum(norm_pdf)
        my_df = pd.DataFrame({
            'log_growth_ratio': x,
            'payout': (np.exp(x * ser['time']) - 1),
            'prob': probabilities,
        })
        return my_df

    def create_prices_for_data(self, df: pd.DataFrame, threshold):
        prices = df.apply(self.create_prices_for_thresholds, thresholds=[threshold], axis=1)
        return prices

    def create_prices_for_thresholds(self, ser: pd.Series, thresholds):
        '''

        :param ser: should have 'vol' values with proportion-based values (not percent values)
        :param thresholds: should have proportion-based values (not percent values)
        :return:
        '''
        distribution = self.create_distribution(ser)
        expected_payouts = dict()
        for threshold in thresholds:
            this_distribution = distribution.copy()
            this_distribution['payout'] -= np.exp(threshold * ser['time']) - 1
            expected_payout = (this_distribution['payout'] > 0) * this_distribution['payout'] * this_distribution['prob']
            expected_payout = expected_payout.sum()
            expected_payouts[threshold] = expected_payout
        return pd.Series(expected_payouts)

    def _static_objective(self, params, cdf_data):
        self.n_iter += 1
        self.set_params(params)
        mu, sigma = self.get_shape_params()
        model_cdfs = [self.excluded_prop + norm.cdf(t, loc=mu, scale=sigma / np.sqrt(d)) for (t, c, d) in cdf_data]
        data_cdfs = [c for (t, c, d) in cdf_data]
        errors = np.array(model_cdfs) - np.array(data_cdfs)
        mse = np.mean(np.power(errors, 2))
        print(params, mse, self.n_iter)
        return mse

    def _dynamic_objective(self, params, data, thresholds):
        self.n_iter += 1
        self.set_params(params)
        prices = data.apply(self.create_prices_for_thresholds, thresholds=thresholds, axis=1)
        earnings = [pd.Series(name=t, data=data['growth'] - np.exp(t * data['time']) + 1) for t in thresholds]
        earnings_df = pd.concat(earnings, axis=1)
        earnings_arr = earnings_df.to_numpy()
        earnings_arr[earnings_arr < 0] = 0
        error = prices.to_numpy() - earnings_arr
        bias = np.mean(error)
        rmse = np.sqrt(np.mean(np.power(error, 2)))
        loss = np.abs(bias) + rmse
        print(params, bias, rmse, loss, self.n_iter)
        return loss

    def get_params(self):
        raw_params = np.array([self.base_sigma, self.excluded_prop, self.vol_var_sigma, self.base_mu])
        return raw_params * self.param_scale

    def _train_static(self, data, thresholds):
        self.n_iter = 0
        log_thresholds = [np.log(1 + t) for t in thresholds]
        data['log_daily_growths'] = np.log(1 + data['growth']) / data['time']
        cdf_points = [(t, percentileofscore(data.loc[data['time'] == d, 'log_daily_growths'], t) / 100, d)
                      for t in log_thresholds for d in data['time'].unique()]
        current_params = self.get_params()
        solution = minimize(self._static_objective, current_params, args=cdf_points,
                            options={'xatol': 0.0005, 'fatol': 0.002},
                            method='Nelder-Mead')
        print()
        self._static_objective(solution.x, cdf_points)
        static_solution = [solution.x[0], solution.x[1], 0, solution.x[3]]
        # test_dynamic_solution = [0.8 * solution.x[0], 0, solution.x[2]]
        # self._dynamic_objective(static_solution, data, thresholds)
        # self._dynamic_objective(test_dynamic_solution, data, thresholds)
        self.set_params(static_solution)

    def train(self, data, thresholds, return_loss=False):
        '''

        :param data: should have 'vol' and 'growth', both as proportion values, not percent.
        :param thresholds: should be a list of proportion values, not percent.
        :return:
        '''
        self._train_static(data, thresholds)
        current_params = self.get_params()
        static_sigma, static_excluded, _, static_mu = tuple(list(current_params))
        average_vol = np.mean(data['vol'])
        vol_ratio = self.base_sigma / average_vol
        best_score = None
        best_try = None
        for j in range(0, 10):
            print()
            factor = 1 - (j/ 10)
            print(f'trying factor = {factor}')
            vol_factor = j/10 * vol_ratio * self.param_scale
            this_try = np.array([factor * static_sigma, static_excluded, vol_factor, static_mu])
            score = self._dynamic_objective(this_try, data, thresholds)
            if best_score is None or score < best_score:
                best_score = score
                best_try = this_try
        initial_guess = best_try
        initial_guess = np.array(initial_guess)
        start = time()
        self.n_iter = 0
        # solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds))
        solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds),
                            options={'fatol': 0.0001, 'maxfev': 1000},
                            method='Nelder-Mead')
        print('training time', time() - start)
        selected_params = solution.x
        print(selected_params, solution.fun)
        self.set_params(selected_params)
        if return_loss:
            return self._dynamic_objective(self.get_params(), data=data, thresholds=thresholds)


class CallPricer:

    @staticmethod
    def expected_value(my_series: pd.Series):
        if pd.isna(my_series['sigma']):
            return None
        x = np.linspace(-0.5, 0.5, 800)
        # power = my_series['power']
        # factor = np.power(np.abs(x), power)
        # x = np.multiply(x, factor)
        norm_pdf = norm.pdf(x, my_series['mu'], my_series['sigma'])
        probabilities = norm_pdf / np.sum(norm_pdf)
        my_df = pd.DataFrame({
            'log_growth_ratio': x,
            'payout': 100 * (np.exp(x) - np.exp(my_series['log_threshold_ratio'])),
            'prob': probabilities,
        })
        my_df['profit'] = (my_df['payout'] > 0) * my_df['payout'] * my_df['prob']
        value = my_df['profit'].sum()
        return value

    def adaptive_objective(self, x: np.array, actual_growths, vol, thresholds: list[float]):
        mu, sigma_0, factor = tuple(x.tolist())
        sigma_0 = np.clip(sigma_0, 0, 1)
        factor = np.clip(factor, 0, 1)
        self.mu = mu
        self.sigma = sigma_0
        self.factor = factor
        # self.power = power
        chunks = []
        for t in thresholds:
            # log_t = np.log(1 + t/100)
            # my_df = pd.DataFrame({
            #     'mu': mu,
            #     'sigma': sigma_0 + factor * vol,
            #     'log_threshold_ratio': log_t
            # })
            # my_df['sigma'] = np.clip(my_df['sigma'], 0.01, 1)
            # prices = my_df.apply(CallPricer.expected_value, axis=1)
            prices = self.calculate_prices(None, t, volatilities=vol)
            returns = (actual_growths > t) * (actual_growths - t)
            net = returns - prices
            results = pd.DataFrame({
                'actuals': actual_growths,
                'prices': prices,
                'returns': returns,
                'net': net,
                'mse': net * net,
            })
            chunks.append(results)
            # print(time() - starting_time)
        joint = pd.concat(chunks, axis=0)
        joint.dropna(inplace=True)
        bias = np.abs(joint['net'].mean())
        mse = np.sqrt(joint['mse'].mean())
        error = bias + mse
        print()
        print(x)
        print(bias, mse, error)
        return error

    # @staticmethod
    # def static_objective(x, log_growths, threshold=None):
    #     if threshold is not None:
    #         threshold = np.log(1 + threshold / 100)
    #     mu, sigma = tuple(x.tolist())
    #     sigma = np.abs(sigma)
    #     if sigma < 0.01:
    #         sigma = 0.01
    #     probabilities = norm.pdf(log_growths, mu, sigma)
    #     if threshold is None:
    #         log_probs = np.log(probabilities)
    #         loss = -1 * np.sum(log_probs)
    #         return loss
    #     my_df = pd.DataFrame({
    #         'log_growth': log_growths,
    #         'prob': probabilities
    #     })
    #     prob_below = norm.cdf(threshold, mu, sigma)
    #     my_df.loc[my_df['log_growth'] < threshold, 'prob'] = prob_below
    #     log_probs = np.log(my_df['prob'])
    #     neg_loss = -1 * np.sum(log_probs)
    #     return neg_loss

    # @staticmethod
    # def solve_static(growths, threshold=None):
    #     log_growths = np.log(1 + growths / 100)
    #     initial_guess = np.array([0, 0.05])
    #     solution = minimize(CallPricer.static_objective, initial_guess, args=(log_growths, threshold),
    #                         method='Nelder-Mead')
    #     return solution.x

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.factor = None
        # self.power = None
        self.volatility_name = None

    def calculate_prices(self, data, threshold, volatilities=None):
        if volatilities is None:
            volatilities = data[self.volatility_name]
        my_df = pd.DataFrame({
            'mu': self.mu,
            'sigma': self.sigma + self.factor * volatilities,
            'log_threshold_ratio': np.log(1 + threshold/100),
        })
        prices = my_df.apply(self.expected_value, axis=1)
        return prices

    def train(self, growths, volatilities, thresholds, volatility_name, return_loss=False, return_solution=False,
              prototype: 'CallPricer' =None):
        self.volatility_name = volatility_name
        # log_growths = np.log(1 + growths / 100)
        initial_guess = [0.02, 0.02, 0.01]
        if prototype is not None:
            initial_guess = [prototype.mu, prototype.sigma, prototype.factor]
        initial_guess = np.array(initial_guess)
        start = time()
        solution = minimize(self.adaptive_objective, initial_guess, args=(growths, volatilities, thresholds),
                            options={'xatol': 0.0005, 'fatol': 0.002},
                            method='Nelder-Mead')
        print()
        print('training time', time() - start)
        x = solution.x
        print(x, solution.fun)
        self.mu = x[0]
        self.sigma = x[1]
        self.factor = x[2]
        # self.power = x[3]
        return_tuple = []
        if return_loss:
            return_tuple.append(solution.fun)
        if return_solution:
            return_tuple.append(x)
        if len(return_tuple) > 0:
            return tuple(return_tuple)