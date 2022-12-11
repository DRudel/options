from scipy.stats import norm, percentileofscore, levy_stable
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from functools import partial
from time import time
from numpy.random import default_rng

my_rng = default_rng()

#decay=0.0035
class StablePricer:
    def __init__(self, lower_z_bound, upper_z_bound, num_partitions, vol_name=None,
                 decay=0, param_scale=10, reg_vol_var_c=0.05, reg_mu=1000, reg_excluded_prop=0.0,
                 reg_alpha=0.0, reg_beta=0, reg_c=0, call=False):
        self.param_scale = param_scale
        self.vol_var_c = 0
        self.mu = 0.0001
        self.alpha = 1.9
        self.base_c = 0.006
        self.beta = 0
        self.n_iter = 0
        self.excluded_prop = 0
        self.decay = decay
        self.reg_vol_var_c = reg_vol_var_c
        self.reg_mu = reg_mu
        self.reg_alpha = reg_alpha
        self.reg_beta = reg_beta
        self.reg_c = reg_c
        self.reg_excluded_prop = reg_excluded_prop
        self.vol_name = vol_name
        self.lower_z_bound = lower_z_bound
        self.upper_z_bound = upper_z_bound
        self.num_partitions = num_partitions
        self._distribution = None
        self._partition = None
        self.call = call
        # self.abstract_distribution = levy_stable(self.alpha, self.beta, self.base_c, self.mu)

    def _get_penalty(self):
        vol_var_penalty = self.reg_vol_var_c * self.vol_var_c ** 2
        mu_penalty = self.reg_mu * self.mu ** 2
        alpha_penalty = self.reg_alpha * (2 - self.alpha) ** 2
        c_penalty = self.reg_c * self.base_c ** 2
        beta_penalty = self.reg_beta * self.beta ** 2
        excluded_penalty = self.reg_excluded_prop * self.excluded_prop ** 2
        total_penalty = vol_var_penalty + mu_penalty + alpha_penalty + c_penalty + beta_penalty + excluded_penalty
        return total_penalty

    def set_distribution(self):
        '''
        Sets the partition and distribution for a single-day's evoluation of the market.
        These values will be multiplied by sigma to determine actual deviation from mean of log-growths.
        :param lower_z:
        :param upper_z:
        :param num_points:
        :return:
        '''
        desired_density = self.num_partitions / (self.upper_z_bound - self.lower_z_bound)
        rv = levy_stable(self.alpha, self.beta)
        width = 1
        if self.call:
            lower_bound = rv.cdf(self.lower_z_bound)
            current_upper_z_bound = self.upper_z_bound
            upper_bound = rv.cdf(current_upper_z_bound)
            while upper_bound < 0.99:
                current_upper_z_bound += 1
                upper_bound = rv.cdf(current_upper_z_bound)
            width = 1 - lower_bound
            print(f'using upper bound of {current_upper_z_bound} and width of {width}')
        else:
            upper_bound = rv.cdf(self.upper_z_bound)
            current_lower_z_bound = self.lower_z_bound
            lower_bound = rv.cdf(current_lower_z_bound)
            while lower_bound > 0.01:
                current_lower_z_bound -= 1
                lower_bound = rv.cdf(current_lower_z_bound)
            width = upper_bound
            print(f"using lower bound of {current_lower_z_bound} and width of {width}.")
        x = np.linspace(self.lower_z_bound, self.upper_z_bound, self.num_partitions)
        density = rv.pdf(x)
        distribution = width * density / np.sum(density)
        self._partition = x
        self._distribution = distribution

    def set_params(self, params, reset_distribution=True):
        for n, field in enumerate(['alpha', 'beta', 'base_c', 'vol_var_c', 'mu', 'excluded_prop']):
            if n < len(params):
                self.__setattr__(field, params[n] / self.param_scale)
        if self.base_c < 0.00001:
            self.base_c = 0.00001
        if self.alpha > 2:
            self.alpha = 2
        if self.alpha < 1:
            self.alpha = 1
        if self.beta > 1:
            self.beta = 1
        if self.beta < -1:
            self.beta = -1
        if self.excluded_prop < 0:
            self.excluded_prop = 0
        if reset_distribution:
            self.set_distribution()

    # def get_shape_params(self, time=1, vol=0):
    #     '''
    #     get_shape_params creates a _miniaturized_ probability distribution equal to 1/time * distribtuion we actually
    #     care about. We care about the distribution (daily_mu) * time with sigma = daily_sigma * sqrt(time), but to
    #     normalize all data to a daily distribution we divide by time to get a distribution with mu = daily mu and
    #     sigma = daily_sigma / sqrt(time)
    #     :param time:
    #     :param vol:
    #     :return:
    #     '''
    #     c = (self.base_c + self.vol_var_c * vol) / np.sqrt(time)
    #     c = np.clip(c, 0.0001, 1)
    #     mu = self.mu
    #     return mu, c

    def create_relative_change_array(self, data_df: pd.DataFrame):
        # This method creates an array of possible payouts at different possible z_scores for base distribution.
        # Note that here we are working _from_ a set partition z_score rather than trying to create one.
        # So effective sigma is the actual expected sigma for the growth after a given time period.
        # This function says nothing about the likelihood of each of these payouts.
        effective_sigmas = (self.base_c + self.vol_var_c * data_df['vol']) * np.power(data_df['time'], 1 / self.alpha)
        # average_effective_sigmas = np.mean(effective_sigmas)
        effective_sigmas = effective_sigmas.to_numpy().reshape(-1, 1)
        z_score_partition= self._partition.reshape(1, -1)
        log_factor_arr = z_score_partition * effective_sigmas
        log_factor_arr = log_factor_arr + data_df['time'].to_numpy().reshape(-1, 1) * self.mu
        relative_change_arr = np.exp(log_factor_arr) - 1
        return relative_change_arr

    def find_expected_relative_payouts(self, data_df: pd.DataFrame, gross_thresholds):
        relative_change = self.create_relative_change_array(data_df)
        if self.call:
            relative_payouts = relative_change - gross_thresholds.to_numpy().reshape(-1, 1)
        else:
            relative_payouts = gross_thresholds.to_numpy().reshape(-1, 1) - relative_change
        option_payouts = (relative_payouts > 0) * relative_payouts
        expected_values = (option_payouts * self._distribution)
        excluded_multiplier = (1 - self.excluded_prop * data_df['time'].to_numpy())
        expected_values = expected_values * excluded_multiplier.reshape(-1, 1)
        expected_relative_payout = expected_values.sum(axis=1)
        return expected_relative_payout

    def find_expected_payouts_from_raw_margin(self, data_df: pd.DataFrame, margin):
        threshold_ser = pd.Series(data=margin/100, index=data_df.index)
        expected_payouts = self.find_expected_relative_payouts(data_df, threshold_ser)
        expected_payouts_ser = pd.Series(data=expected_payouts, index=data_df.index)
        return expected_payouts_ser

    def create_prices_for_thresholds(self, data_df: pd.DataFrame, log_daily_thresholds):
        expected_payouts = dict()
        for threshold in log_daily_thresholds:
            gross_thresholds = np.exp(data_df['time'] * threshold) - 1
            expected_payouts[threshold] = self.find_expected_relative_payouts(data_df, gross_thresholds=gross_thresholds)
        return pd.DataFrame(expected_payouts)

    def _static_objective(self, params, cdf_data):
        self.n_iter += 1
        self.set_params(params, reset_distribution=False)
        alpha, beta, base_c, mu = self.alpha, self.beta, self.base_c, self.mu
        #base_distribution = levy_stable(alpha, beta, c, mu)
        # model_cdfs = [norm.cdf(t, loc=mu, scale=base_c / np.power(d, 1 / self.alpha))
        #               for (t, c, d) in cdf_data]
        model_cdfs = [self.excluded_prop * d + (1 - self.excluded_prop * d) *
                      levy_stable.cdf(t, alpha=alpha, beta=beta, loc=mu, scale=base_c / np.power(d, 1 / self.alpha))
                      for (t, c, d) in cdf_data]
        data_cdfs = [c for (t, c, d) in cdf_data]
        errors = np.array(model_cdfs) - np.array(data_cdfs)
        mse = np.mean(np.power(errors, 2))
        print(params, mse, self.n_iter)
        return mse

    def _dynamic_objective(self, params, data, log_daily_thresholds, show_values=False):
        self.n_iter += 1
        self.set_params(params)
        # prices = data.apply(self.create_prices_for_thresholds, thresholds=thresholds, axis=1)
        prices = self.create_prices_for_thresholds(data_df=data, log_daily_thresholds=log_daily_thresholds)
        earnings = [pd.Series(name=t, data=data['growth'] - np.exp(t * data['time']) + 1) for t in log_daily_thresholds]
        earnings_df = pd.concat(earnings, axis=1)
        if not self.call:
            earnings_df[:] = -1 * earnings_df[:]
        earnings_df[earnings_df < 0] = 0
        earnings_arr = earnings_df.to_numpy()
        # earnings_arr[earnings_arr < 0] = 0
        error = prices.to_numpy() - earnings_arr
        error_df = pd.DataFrame(data=error, columns=prices.columns)
        # decay_df indicates how much we down_weight data from the past.
        decay_df = data[['order']].copy()
        decay_df['factor'] = np.power(1 / (1 - self.decay), decay_df['order'])
        decay_df.reset_index(inplace=True)
        decay_df['factor'] = decay_df['factor'] / decay_df['factor'].mean()
        decayed_error_df = error_df.mul(decay_df['factor'], axis=0)
        threshold_level_errors = decayed_error_df.mean(axis=0)
        threshold_level_absolute_average_bias = np.mean(np.abs(threshold_level_errors))
        decayed_error = decayed_error_df.to_numpy()
        # if show_values:
        #     data.to_csv('growth_data.csv')
        #     prices.to_csv('price_data.csv')
        #     earnings_df.to_csv('earnings_data.csv')
        #     error_df.to_csv('error.csv')
        bias = np.mean(decayed_error)
        rmse = np.sqrt(np.mean(np.power(decayed_error, 2)))
        penalty = self._get_penalty()
        loss = 100 * np.abs(bias) + 10 * threshold_level_absolute_average_bias + rmse + penalty
        if show_values:
            print(params, 100 * bias, 10 * threshold_level_absolute_average_bias, rmse, penalty, loss, self.n_iter)
        return loss

    def get_params(self):
        raw_params = np.array([self.alpha, self.beta, self.base_c, self.vol_var_c, self.mu, self.excluded_prop])
        return raw_params * self.param_scale

    def _train_static(self, data, log_daily_thresholds):
        self.n_iter = 0
        data['log_daily_growths'] = np.log(1 + data['growth']) / data['time']
        cdf_points = [(t, percentileofscore(data.loc[data['time'] == d, 'log_daily_growths'], t) / 100, d)
                      for t in log_daily_thresholds for d in data['time'].unique()]
        current_params = self.get_params()
        solution = minimize(self._static_objective, current_params, args=cdf_points,
                            options={'xatol': 0.0005, 'fatol': 0.001},
                            bounds=[(10, 20), (-10, None), (0.001, None), (0, 0), (None, None), (0, None)],
                            method='Nelder-Mead')
        print()
        loss = self._static_objective(solution.x, cdf_points)
        static_solution = [solution.x[0], solution.x[1], solution.x[2], 0, solution.x[4], solution.x[5]]
        # test_dynamic_solution = [0.8 * solution.x[0], 0, solution.x[2]]
        # self._dynamic_objective(static_solution, data, thresholds)
        # self._dynamic_objective(test_dynamic_solution, data, thresholds)
        self.set_params(static_solution)
        return loss

    def train(self, data, thresholds, return_loss=False, rough=False):
        '''
        :param data: should have 'vol' and 'growth', both as proportion values, not percent.
        :param thresholds: should be a list of proportion values, not percent.
        :return:
        '''
        log_thresholds = [np.log(1 + t) for t in thresholds]
        static_loss = self._train_static(data, log_daily_thresholds=log_thresholds)
        if rough:
            return static_loss
        current_params = self.get_params()
        static_alpha, static_beta, static_c, _, static_mu, static_excluded_prop = tuple(list(current_params))
        # static_excluded = np.clip(static_excluded, 0.0015, 0.003)
        average_vol = np.mean(data['vol'])
        vol_ratio = self.base_c / average_vol
        best_score = None
        best_try = None
        for j in range(6, 7):
            print()
            print()
            factor = 1 - (j/ 10)
            print(f'trying factor = {factor}')
            vol_factor = j/10 * vol_ratio * self.param_scale
            this_try = np.array([static_alpha, static_beta, factor * static_c, vol_factor, static_mu,
                                 static_excluded_prop])
            solution = minimize(self._dynamic_objective, this_try, args=(data, log_thresholds, True),
                                # options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
                                options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
                                bounds=[(10, 20), (-10, 10), (0.00001, None), (0, None), (None, None), (0, None)],
                                method='Nelder-Mead')
            selected_params = solution.x
            print('selected parameters:', selected_params)
            print('loss = ', solution.fun)
            # score = self._dynamic_objective(this_try, data, log_daily_thresholds=log_thresholds, show_values=True)
            # if best_score is None or score < best_score:
            #     best_score = score
            #     best_try = this_try
        # initial_guess = best_try
        # initial_guess = np.array(initial_guess)
        # start = time()
        # self.n_iter = 0
        # # solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds))
        # # solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds),
        # #                     method='COBYLA', options={'rhobeg': 0.5})
        # solution = minimize(self._dynamic_objective, initial_guess, args=(data, log_thresholds, True),
        #                     # options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
        #                     options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
        #                     bounds=[(10, 20), (-10, 10), (0.00001, None), (0, None), (None, None), (0, None)],
        #                     method='Nelder-Mead')
        # print('training time', time() - start)
        # selected_params = solution.x
        # print('selected parameters:', selected_params, solution.fun)
        # self.set_params(selected_params)
        # if return_loss:
        #     return self._dynamic_objective(self.get_params(), data=data, log_daily_thresholds=log_thresholds,
        #                                    show_values=True)


class NormCallPricer:
    def __init__(self, lower_z_bound, upper_z_bound, num_partitions, vol_name=None, decay=0.0035, param_scale=10,
                 reg_vol_var_sigma=0.05, reg_base_mu=1000, reg_excluded_prop=0.0):
        self.param_scale = param_scale
        self.base_sigma = 0.006
        self.vol_var_sigma = 0
        self.base_mu = 0.0001
        self.excluded_prop = 0.0002
        self.n_iter = 0
        self.decay = decay
        self.reg_vol_var_sigma = reg_vol_var_sigma
        self.reg_base_mu = reg_base_mu
        self.reg_excluded_prop = reg_excluded_prop
        self.vol_name = vol_name
        self.lower_z_bound = lower_z_bound
        self.upper_z_bound = upper_z_bound
        self.num_partitions = num_partitions
        self._distribution = None
        self._partition = None
        self.set_distribution()

    def _get_penalty(self):
        vol_var_penalty = self.reg_vol_var_sigma * self.vol_var_sigma ** 2
        base_mu_penalty = self.reg_base_mu * self.base_mu ** 2
        excluded_penalty = self.reg_excluded_prop * self.excluded_prop ** 2
        total_penalty = vol_var_penalty + base_mu_penalty + excluded_penalty
        return total_penalty

    def set_distribution(self):
        '''
        sets up a re-usable distribution of across z-scores ranging from lower_z to upper_z. Completely abstract
        distribution, essentially an interval of a distribution. No parameters.
        These values will be multiplied by sigma to determine actual deviation from mean of log-growths.
        Currently using normal curve.
        :param lower_z:
        :param upper_z:
        :param points:
        :return:
        '''
        x = np.linspace(self.lower_z_bound, self.upper_z_bound, self.num_partitions)
        below_bound = norm.cdf(self.lower_z_bound)
        density = norm.pdf(x)
        distribution = (1 - below_bound) * density / np.sum(density)
        self._partition = x
        self._distribution = distribution

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
            self.excluded_prop = 0

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
        mu = self.base_mu / time
        return mu, sigma

    def create_payout_array(self, data_df: pd.DataFrame):
        # This method creates an array of possible payouts at different possible z_scores for base distribution.
        # Note that here we are working _from_ a set partition z_score rather than trying to create one.
        # So effective sigma is the actual expected sigma for the growth after a given time period.
        # This function says nothing about the likelihood of each of these payouts.
        effective_sigmas = (self.base_sigma + self.vol_var_sigma * data_df['vol']) * np.sqrt(data_df['time'])
        # average_effective_sigmas = np.mean(effective_sigmas)
        effective_sigmas = effective_sigmas.to_numpy().reshape(-1, 1)
        z_score_partition = self._partition.reshape(1, -1)
        growth_arr = z_score_partition * effective_sigmas
        growth_arr = growth_arr + data_df['time'].to_numpy().reshape(-1, 1) * self.base_mu
        payout_arr = np.exp(growth_arr) - 1
        return payout_arr

    def find_expected_payouts_from_raw_margin(self, data_df: pd.DataFrame, margin):
        threshold_ser = pd.Series(data=margin/100, index=data_df.index)
        expected_payouts = self.find_expected_payouts(data_df, threshold_ser)
        expected_payouts_ser = pd.Series(data=expected_payouts, index=data_df.index)
        return expected_payouts_ser

    def find_expected_payouts(self, data_df: pd.DataFrame, gross_thresholds):
        closing_payouts = self.create_payout_array(data_df)
        relative_payouts = closing_payouts - gross_thresholds.to_numpy().reshape(-1, 1)
        option_payouts = (relative_payouts > 0) * relative_payouts
        expected_values = (option_payouts * self._distribution)
        excluded_multiplier = (1 - self.excluded_prop * data_df['time'].to_numpy())
        expected_values = expected_values * excluded_multiplier.reshape(-1, 1)
        expected_payout = expected_values.sum(axis=1)
        return expected_payout

    def create_prices_for_thresholds(self, data_df: pd.DataFrame, log_daily_thresholds):
        expected_payouts = dict()
        for threshold in log_daily_thresholds:
            gross_thresholds = np.exp(data_df['time'] * threshold) - 1
            expected_payouts[threshold] = self.find_expected_payouts(data_df, gross_thresholds=gross_thresholds)
        return pd.DataFrame(expected_payouts)

    def _static_objective(self, params, cdf_data):
        self.n_iter += 1
        self.set_params(params)
        mu, sigma = self.get_shape_params()
        model_cdfs = [self.excluded_prop * d +
                      (1 - self.excluded_prop * d) * norm.cdf(t, loc=mu, scale=sigma / np.sqrt(d)) for (t, c, d) in cdf_data]
        data_cdfs = [c for (t, c, d) in cdf_data]
        errors = np.array(model_cdfs) - np.array(data_cdfs)
        bias = np.mean(errors)
        mse = np.mean(np.power(errors, 2))
        print(params, mse, self.n_iter)
        return mse

    def _dynamic_objective(self, params, data, log_daily_thresholds, show_values=False):
        self.n_iter += 1
        self.set_params(params)
        # prices = data.apply(self.create_prices_for_thresholds, thresholds=thresholds, axis=1)
        prices = self.create_prices_for_thresholds(data_df=data, log_daily_thresholds=log_daily_thresholds)
        earnings = [pd.Series(name=t, data=data['growth'] - np.exp(t * data['time']) + 1) for t in log_daily_thresholds]
        earnings_df = pd.concat(earnings, axis=1)
        earnings_df[earnings_df < 0] = 0
        earnings_arr = earnings_df.to_numpy()
        # earnings_arr[earnings_arr < 0] = 0
        error = prices.to_numpy() - earnings_arr
        error_df = pd.DataFrame(data=error, columns=prices.columns)

        # decay_df indicates how much we down_weight data from the past.
        decay_df = data[['order']].copy()
        decay_df['factor'] = np.power(1 / (1 - self.decay), decay_df['order'])
        decay_df.reset_index(inplace=True)
        decay_df['factor'] = decay_df['factor'] / decay_df['factor'].mean()
        decayed_error_df = error_df.mul(decay_df['factor'], axis=0)
        threshold_level_errors = decayed_error_df.mean(axis=0)
        threshold_level_absolute_average_bias = np.mean(np.abs(threshold_level_errors))
        decayed_error = decayed_error_df.to_numpy()
        # if show_values:
        #     data.to_csv('growth_data.csv')
        #     prices.to_csv('price_data.csv')
        #     earnings_df.to_csv('earnings_data.csv')
        #     error_df.to_csv('error.csv')
        bias = np.mean(decayed_error)
        rmse = np.sqrt(np.mean(np.power(decayed_error, 2)))
        penalty = self._get_penalty()
        loss = np.abs(bias) + 3 * threshold_level_absolute_average_bias + rmse + penalty
        if show_values:
            print(params, bias, threshold_level_absolute_average_bias, rmse, penalty, loss, self.n_iter)
        return loss

    def get_params(self):
        raw_params = np.array([self.base_sigma, self.excluded_prop, self.vol_var_sigma, self.base_mu])
        return raw_params * self.param_scale

    def _train_static(self, data, log_daily_thresholds):
        self.n_iter = 0
        data['log_daily_growths'] = np.log(1 + data['growth']) / data['time']
        cdf_points = [(t, percentileofscore(data.loc[data['time'] == d, 'log_daily_growths'], t) / 100, d)
                      for t in log_daily_thresholds for d in data['time'].unique()]
        current_params = self.get_params()
        solution = minimize(self._static_objective, current_params, args=cdf_points,
                            bounds=[(0, None), (0, 0), (0, 0), (None, None)],
                            options={'xatol': 0.0005, 'fatol': 0.001},
                            method='Nelder-Mead')
        print()
        loss = self._static_objective(solution.x, cdf_points)
        static_solution = [solution.x[0], solution.x[1], 0, solution.x[3]]
        # test_dynamic_solution = [0.8 * solution.x[0], 0, solution.x[2]]
        # self._dynamic_objective(static_solution, data, thresholds)
        # self._dynamic_objective(test_dynamic_solution, data, thresholds)
        self.set_params(static_solution)
        return loss

    def train(self, data, thresholds, return_loss=False, rough=False):
        '''
        :param data: should have 'vol' and 'growth', both as proportion values, not percent.
        :param thresholds: should be a list of proportion values, not percent.
        :return:
        '''
        log_thresholds = [np.log(1 + t) for t in thresholds]
        static_loss = self._train_static(data, log_daily_thresholds=log_thresholds)
        if rough:
            return static_loss
        current_params = self.get_params()
        static_sigma, static_excluded, _, static_mu = tuple(list(current_params))
        # static_excluded = np.clip(static_excluded, 0.0015, 0.003)
        average_vol = np.mean(data['vol'])
        vol_ratio = self.base_sigma / average_vol
        best_score = None
        best_try = None
        for j in range(0, 1):
            print()
            factor = 1 - (j/ 10)
            print(f'trying factor = {factor}')
            vol_factor = j/10 * vol_ratio * self.param_scale
            this_try = np.array([factor * static_sigma, static_excluded, vol_factor, static_mu])
            score = self._dynamic_objective(this_try, data, log_daily_thresholds=log_thresholds)
            if best_score is None or score < best_score:
                best_score = score
                best_try = this_try
        initial_guess = best_try
        initial_guess = np.array(initial_guess)
        start = time()
        self.n_iter = 0
        # solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds))
        # solution = minimize(self._dynamic_objective, initial_guess, args=(data, thresholds),
        #                     method='COBYLA', options={'rhobeg': 0.5})
        solution = minimize(self._dynamic_objective, initial_guess, args=(data, log_thresholds, True),
                            # options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
                            options={'fatol': 0.00002, 'xatol': 0.0002, 'maxfev': 1000},
                            bounds=[(0, None), (0, 0), (0, 0), (None, None)],
                            method='Nelder-Mead')
        print('training time', time() - start)
        selected_params = solution.x
        print('selected parameters:', selected_params, solution.fun)
        self.set_params(selected_params)
        if return_loss:
            return self._dynamic_objective(self.get_params(), data=data, log_daily_thresholds=log_thresholds,
                                           show_values=True)


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