import pandas as pd
from numpy.random import default_rng
from scipy import stats

my_rng = default_rng()

VOLATILITY_LIST = list(range(1, 9)) + [12, 36]
VOLATILITY_PAIRS = ((6, 3), (3, 1), (6, 1))
# VOLATILITY_NAMES = {x:'vol_' + str(x) for x in VOLATILITY_LIST}
# VOLATILITY_DICT = {y: x for (x,y) in VOLATILITY_NAMES.items()}

EXP_VOLATILITY_ALPHAS = [0.2, 0.25, 0.30, 0.35, 0.4]

PRICING_VOLATILITIES = ['vol_6', 'vol_4', 'vol_12', 'evol_30', 'evol_25', 'evol_40', 'evol_35']

# PRICING_VOLATILITIES = list(range(2, 9)) + [12]

CP_TRENDS = list(range(2, 8))
PRICE_TRENDS = list(range(2, 7))
UNEMPLOYMENT_TRENDS = list(range(2, 7))

# GROWTH_DICT = {
#     'three_month_change': 3,
#     'four_month_change': 4,
# }

GROWTH_DICT = dict(
    {
    'one_month_change': 1,
    'two_month_change': 2,
    'three_month_change': 3,
    'four_month_change': 4,
    'five_month_change': 5,
    'six_month_change': 6,
    'seven_month_change': 7,
    # 'nine_month_change': 9,
    # 'year_change': 12,
     }
)

GROWTH_NAMES = {x: y for (y, x) in GROWTH_DICT.items()}

NON_FEATURES = ['date', 'price', 'index', 'lm_cpi', 'price_36_max', 'price_36_min', 'price_36_spread']
NON_FEATURES.extend(list(GROWTH_DICT.keys()))
marginal_features = ['ue_rolling_4', 'fed_trend_diff', 'cpi_rolling_4', 'vol_7']


def calc_downstream_returns(my_df, num_months):
    future_prices = my_df['price'].shift(-1 * num_months)
    returns = 100 * (future_prices / my_df['price'] - 1)
    return returns


def calc_trend_diff(my_series, num_periods):
    return my_series - my_series.rolling(num_periods).mean()


def calc_avg_abs_change(my_series, num_periods):
    ratios = 100 * (my_series - my_series.shift(1)) / my_series.shift(1)
    return ratios.abs().rolling(num_periods).mean()

def calc_ewb_abs_change(my_series, alpha):
    ratios = 100 * (my_series - my_series.shift(1)) / my_series.shift(1)
    return ratios.abs().ewm(alpha=alpha).mean()


def calc_price_percentile(my_roller):
    return stats.percentileofscore(my_roller, my_roller[-1])


def calc_final_value(change, threshold):
    return_values = change.copy() - threshold
    return_values[return_values < 0] = 0
    return return_values

def create_labels(my_data):
    label_dict = {tp: calc_downstream_returns(my_data, GROWTH_DICT[tp]) for tp in GROWTH_DICT}
    return pd.DataFrame(label_dict)


def prepare_data(my_data):
    my_df = my_data.copy()
    my_df['cpi_change'] = my_data['lm_cpi'] / my_data['lm_cpi'].shift(12)
    my_df['cpi_rolling_4'] = my_df['cpi_change'].rolling(4, min_periods=4).mean()
    for num_months in CP_TRENDS:
        my_df['cpi_trend_diff_' + str(num_months)] = calc_trend_diff(my_df['cpi_change'], num_months)
    my_df['fed_rate_change'] = my_data['lm_fed_rate'] - my_data['lm_fed_rate'].shift(1)
    my_df['fed_trend_diff'] = calc_trend_diff(my_df['lm_fed_rate'], 4)
    my_df['ue_rolling_4'] = my_data['lm_unemployment'].rolling(4).mean()
    for num_months in UNEMPLOYMENT_TRENDS:
        my_df['ue_trend_diff_' + str(num_months)] = calc_trend_diff(my_df['lm_unemployment'], num_months)
    my_df['price_36_max'] = my_data['price'].rolling(36).max()
    my_df['price_36_min'] = my_data['price'].rolling(36).min()
    my_df['price_36_spread'] = my_df['price_36_max'] - my_df['price_36_min']
    my_df['price_36_prank'] = (my_data['price'] - my_df['price_36_min']) / my_df['price_36_spread']
    my_df['price_36_pspread'] = my_df['price_36_spread'] / my_df['price_36_max']
    my_df['price_off_high'] = 1 - my_data['price'] / my_df['price_36_max']
    my_df['price_36_percentile'] = my_data['price'].rolling(36).apply(calc_price_percentile, raw=True)
    for num_months in VOLATILITY_LIST:
        label = 'vol_' + str(num_months)
        my_df[label] = calc_avg_abs_change(my_data['price'], num_months)

    for (m, n) in VOLATILITY_PAIRS:
        label = 'vol_diff_' + str(m) + '_' + str(n)
        my_df[label] = my_df['vol_' + str(m)] - my_df['vol_' + str(n)]

    for alpha in EXP_VOLATILITY_ALPHAS:
        label = 'evol_' + str(int(100 * alpha))
        my_df[label] = calc_ewb_abs_change(my_data['price'], alpha)

    for num_months in PRICE_TRENDS:
        my_df['price_trend_' + str(num_months)] = calc_trend_diff(my_df['price'], num_months)
    for time_period in GROWTH_DICT:
        my_df[time_period] = calc_downstream_returns(my_df, GROWTH_DICT[time_period])
    return my_df
