import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy import stats
import pickle
import datetime as dt

trends = range(2, 7)

GROWTH_DICT = dict(
    {
    'one_month_change': 30,
    'two_month_change': 60,
    'three_month_change': 91,
    'four_month_change': 121,
    'five_month_change': 152,
    'six_month_change': 183,
    'seven_month_change': 213,
     }
)

GROWTH_NAMES = {x: y for (y, x) in GROWTH_DICT.items()}
NON_FEATURES = ['date', 'price', 'index', 'lm_cpi', 'price_36_max', 'price_36_min', 'price_36_spread']
NON_FEATURES.extend(list(GROWTH_DICT.keys()))


def calc_downstream_returns(my_df, num_days):
    future_prices = my_df['close'].shift(-1 * num_days)
    returns = 100 * (future_prices / my_df['close'] - 1)
    return returns

def calc_trend_diff(my_series, num_periods):
    return my_series - my_series.rolling(num_periods).mean()

def prepare_cpi_data(cpi_ser, offset='43D'):
    return_df = pd.DataFrame(
        {
            'cpi_annual_change': cpi_ser / cpi_ser.shift(12) - 1,
            'cpi_monthly_change': cpi_ser / cpi_ser.shift(1) - 1
        }
    )
    return_df['cpi_annual_rolling_4'] = return_df['cpi_annual_change'].rolling(4, min_periods=4).mean()
    return_df['cpi_monthly_rolling_4'] = return_df['cpi_monthly_change'].rolling(4, min_periods=4).mean()
    return_df['cpi_annual_trend_diff_4'] = calc_trend_diff(return_df['cpi_annual_change'], 4)
    return_df['cpi_monthly_trend_diff_4'] = calc_trend_diff(return_df['cpi_monthly_change'], 4)
    return_df['data_date'] = return_df.index.copy() + pd.Timedelta(offset)
    return return_df


def prepare_fed_funds_data(fed_funds_ser, offset='32D'):
    return_df = pd.DataFrame(
        {
            'fed_funds_rate': fed_funds_ser,
            'fed_funds_change': fed_funds_ser - fed_funds_ser.shift(1),
            'fed_trend_diff_4': calc_trend_diff(fed_funds_ser, 4),
            'data_date': fed_funds_ser.index.copy() + pd.Timedelta(offset)
        }
    )
    return return_df


def prep_unemployment_data(unemployment_ser, offset='33D'):
    return_df = pd.DataFrame(
        {
            'ue_rate': unemployment_ser,
            'ue_rate_rolling_4': unemployment_ser.rolling(4).mean(),
            'ue_rate_trend_diff_4': calc_trend_diff(unemployment_ser, 4),
            'data_date': unemployment_ser.index.copy() + pd.Timedelta(offset)
        }
    )
    return return_df


def process_price_data(price_df):
    env_3y = price_envelope_stats(price_df, (3 * 365 + 1))
    env_1y = price_envelope_stats(price_df, 365)
    env_data = env_1y.join(env_3y)
    return env_data


def price_envelope_stats(price_df, num_days):
    time_offset = str(num_days) + 'D'
    return_df = pd.DataFrame(
        {
            'price_env_max': price_df['close'].rolling(time_offset).max(),
            'price_env_min': price_df['close'].rolling(time_offset).min()
        }
    )
    return_df['price_env_spread'] = return_df['price_env_max'] - return_df['price_env_min']
    return_df['price_env_prank'] = (price_df['close'] - return_df['price_env_min']) / return_df['price_env_spread']
    return_df['price_env_pspread'] = return_df['price_env_spread'] / return_df['price_env_max']
    return_df['price_env_off_high'] = 1 - price_df['close'] / return_df['price_env_max']
    return_df['price_env_percentile'] = price_df['close'].rolling(time_offset).apply(calc_price_percentile, raw=True)
    return_df.drop(['price_env_spread', 'price_env_min', 'price_env_max'], axis=1, inplace=True)
    return_df = return_df.add_suffix('_' + time_offset)
    return return_df


def calc_price_percentile(my_roller):
    return stats.percentileofscore(my_roller, my_roller[-1])


def ema_relative_price_difference(price_df: pd.DataFrame, neg_log_alpha):
    alpha = np.exp(-1 * neg_log_alpha)
    data = price_df[['close']].copy()
    data['ema'] = data['close'].ewm(alpha=alpha).mean()
    data['abs_gap'] = data['close'] - data['ema']
    data['rel_gap'] = data['abs_gap'] / data['close']
    data['rel_gap_ema'] = data['rel_gap'].ewm(alpha=alpha).mean()
    data['rel_delta'] = data['rel_gap'] - data['rel_gap_ema']
    data.drop(['close', 'ema', 'abs_gap'], axis=1, inplace=True)
    data = data.add_suffix(int(neg_log_alpha * 10))
    return data


def price_envelope_trend_stats(price_df, num_days):
    time_offset = str(num_days) + 'D'
    return_df = pd.DataFrame(
        {
            'price_env_average': price_df['close'].rolling(time_offset).mean(),
            'price_env_std': price_df['close'].rolling(time_offset).std(),
            'price_env_delta': price_df['close'].rolling(time_offset).agg(lambda x: x[-1] - x[0])
        }
    )
    return_df['price_env_trend_diff'] = (price_df['close'] - return_df['price_env_average']) / price_df['close']
    return_df['price_env_acc'] = 2 * return_df['price_env_trend_diff'] - \
                                 (return_df['price_env_delta'] / price_df['close'])
    return_df['price_env_trend_strength'] = return_df['price_env_delta'] / \
                                            np.sqrt((return_df['price_env_std'] * return_df['price_env_average']))
    return_df.drop(['price_env_average', 'price_env_std', 'price_env_delta'], axis=1, inplace=True)
    return_df = return_df.add_suffix('_' + time_offset)
    return return_df


def generate_vol_data(price_df, vol_granularity, neg_log_alpha):
    alpha = np.exp(-1 * neg_log_alpha)
    data = price_df[['close']].copy()
    vol_gran_code = str(vol_granularity) + 'd'
    data['abs_interval_change'] = data.rolling(vol_gran_code).agg(lambda x: abs(x.iloc[-1] - x.iloc[0]))
    data = data.iloc[vol_granularity:]
    data['rel_interval_change'] = data['abs_interval_change'] / price_df['close'].rolling(vol_gran_code).mean()
    data['square_interval_change'] = np.power(data['rel_interval_change'], 2)
    data['decayed_ms_change'] = data['square_interval_change'].ewm(alpha=alpha).mean()
    data['evol'] = np.sqrt(data['decayed_ms_change'])
    return_data = data[['evol']].copy()
    return_data = return_data.add_suffix('_' + vol_gran_code + '_' + str(int(10 * neg_log_alpha)))
    return return_data


def generate_period_data():
    today = dt.datetime.today()
    cpi_data: pd.DataFrame = pickle.load(open('cpi_data.pickle', 'rb')).set_index('data_date', drop=True)
    ue_data: pd.DataFrame = pickle.load(open('ue_data.pickle', 'rb')).set_index('data_date', drop=True)
    ffund_data: pd.DataFrame = pickle.load(open('ffund_data.pickle', 'rb')).set_index('data_date', drop=True)
    joint_period_data = pd.concat([cpi_data, ffund_data, ue_data], axis=1)
    joint_period_data.loc[today] = None
    joint_period_data = joint_period_data.resample('1d').mean()
    return joint_period_data


def generate_price_features(price_data, num_days_trim=1096):
    base_price_data = process_price_data(price_data)
    ema_price_data_chunks = []
    for nla in [2.5, 3, 3.5]:
        ema_price_data_chunks.append(ema_relative_price_difference(price_data, nla))
    ema_price_data = pd.concat(ema_price_data_chunks, axis=1)

    window_trend_data_chunks = []
    for window_width in [45, 110]:
        window_trend_data_chunks.append(price_envelope_trend_stats(price_data, window_width))
    window_trend_data = pd.concat(window_trend_data_chunks, axis=1)

    vol_data_chunks = []
    for vol_nla in [2.5, 3, 3.5]:
        for vol_granularity in [7, 15, 21]:
            vol_data_chunks.append(generate_vol_data(price_data, vol_granularity, vol_nla))
    vol_data = pd.concat(vol_data_chunks, axis=1)

    raw_close = price_data[['close']].copy()
    joint_price_data = raw_close.join(base_price_data).join(ema_price_data).join(window_trend_data).join(vol_data)
    joint_price_data = joint_price_data.dropna()
    joint_price_data = joint_price_data.resample('1d').mean()
    joint_price_data = joint_price_data.fillna(method='ffill')
    joint_price_data = joint_price_data.iloc[num_days_trim:]
    for growth_name, num_days in GROWTH_DICT.items():
        joint_price_data[growth_name] = calc_downstream_returns(joint_price_data, num_days)
    return joint_price_data


def generate_full_data(price_data, **kwargs):
    price_features = generate_price_features(price_data, **kwargs)
    period_features = generate_period_data().fillna(method='ffill')
    return_data = price_features.join(period_features)
    return return_data