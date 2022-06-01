import pandas as pd
from numpy.random import default_rng
from scipy import stats

my_rng = default_rng()

VOLATILITY_LIST = list(range(1, 9)) + [12, 36]
VOLATILITY_PAIRS = ((6, 3), (3, 1), (6, 1))
VOLATILITY_NAMES = {x:'vol_' + str(x) for x in VOLATILITY_LIST}
VOLATILITY_DICT = {y: x for (x,y) in VOLATILITY_NAMES.items()}

CP_TRENDS = list(range(2, 8))
PRICE_TRENDS = list(range(2, 7))
UNEMPLOYMENT_TRENDS = list(range(2, 7))

# GROWTH_DICT = {
#     'three_month_change': 3,
#     'four_month_change': 4,
# }

GROWTH_DICT = dict(
    {
    'two_month_change': 2,
    'three_month_change': 3,
    'four_month_change': 4,
    'five_month_change': 5,
    'six_month_change': 6,
    'seven_month_change': 7,
    'nine_month_change': 9,
    'year_change': 12,
     }
)

GROWTH_NAMES = {x: y for (y, x) in GROWTH_DICT.items()}

NON_FEATURES = ['date', 'price', 'index', 'cpi', 'price_36_max', 'price_36_min', 'price_36_spread']
NON_FEATURES.extend(list(GROWTH_DICT.keys()))


def calc_downstream_returns(my_df, num_months):
    future_prices = my_df['price'].shift(-1 * num_months)
    returns = 100 * (future_prices / my_df['price'] - 1)
    return returns


def calc_trend_diff(my_series, num_periods):
    return my_series - my_series.rolling(num_periods).mean()


def calc_avg_abs_change(my_series, num_periods):
    ratios = 100 * (my_series - my_series.shift(1)) / my_series
    return ratios.abs().rolling(num_periods).mean()


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
    my_df['cpi_change'] = my_data['cpi'] / my_data['cpi'].shift(12)
    my_df['cpi_rolling_4'] = my_df['cpi_change'].rolling(4, min_periods=4).mean()
    for num_months in CP_TRENDS:
        my_df['cpi_trend_diff_' + str(num_months)] = calc_trend_diff(my_df['cpi_change'], num_months)
    # my_df['cpi_trend_diff_2'] = calc_trend_diff(my_df['cpi_change'], 2)
    # my_df['cpi_trend_diff_3'] = calc_trend_diff(my_df['cpi_change'], 3)
    # my_df['cpi_trend_diff_4'] = calc_trend_diff(my_df['cpi_change'], 4)
    # my_df['cpi_trend_diff_5'] = calc_trend_diff(my_df['cpi_change'], 5)
    # my_df['cpi_trend_diff_6'] = calc_trend_diff(my_df['cpi_change'], 6)
    # my_df['cpi_trend_diff_7'] = calc_trend_diff(my_df['cpi_change'], 7)
    my_df['fed_rate_change'] = my_data['fed_rate'] - my_data['fed_rate'].shift(1)
    my_df['fed_trend_diff'] = calc_trend_diff(my_df['fed_rate'], 4)
    my_df['ue_rolling_4'] = my_data['unemployment'].rolling(4).mean()
    for num_months in UNEMPLOYMENT_TRENDS:
        my_df['ue_trend_diff_' + str(num_months)] = calc_trend_diff(my_df['unemployment'], num_months)
    # my_df['ue_trend_diff_2'] = calc_trend_diff(my_df['unemployment'], 2)
    # my_df['ue_trend_diff_3'] = calc_trend_diff(my_df['unemployment'], 3)
    # my_df['ue_trend_diff_4'] = calc_trend_diff(my_df['unemployment'], 4)
    # my_df['ue_trend_diff_5'] = calc_trend_diff(my_df['unemployment'], 5)
    # my_df['ue_trend_diff_6'] = calc_trend_diff(my_df['unemployment'], 6)
    my_df['price_36_max'] = my_data['price'].rolling(36).max()
    my_df['price_36_min'] = my_data['price'].rolling(36).min()
    my_df['price_36_spread'] = my_df['price_36_max'] - my_df['price_36_min']
    my_df['price_36_prank'] = (my_data['price'] - my_df['price_36_min']) / my_df['price_36_spread']
    my_df['price_36_pspread'] = my_df['price_36_spread'] / my_df['price_36_max']
    my_df['price_off_high'] = 1 - my_data['price'] / my_df['price_36_max']
    my_df['price_36_percentile'] = my_data['price'].rolling(36).apply(calc_price_percentile, raw=True)
    for num_months in VOLATILITY_LIST:
        my_df[VOLATILITY_NAMES[num_months]] = calc_avg_abs_change(my_data['price'], num_months)
    # my_df['vol_36'] = calc_avg_abs_change(my_data['price'], 36)
    # my_df['vol_12'] = calc_avg_abs_change(my_data['price'], 12)
    # my_df['vol_8'] = calc_avg_abs_change(my_data['price'], 8)
    # my_df['vol_7'] = calc_avg_abs_change(my_data['price'], 7)
    # my_df['vol_6'] = calc_avg_abs_change(my_data['price'], 6)
    # my_df['vol_5'] = calc_avg_abs_change(my_data['price'], 5)
    # my_df['vol_4'] = calc_avg_abs_change(my_data['price'], 4)
    # my_df['vol_3'] = calc_avg_abs_change(my_data['price'], 3)
    # my_df['vol_2'] = calc_avg_abs_change(my_data['price'], 2)
    # my_df['vol_1'] = calc_avg_abs_change(my_data['price'], 1)
    for (m, n) in VOLATILITY_PAIRS:
        label = 'vol_diff_' + str(m) + '_' + str(n)
        my_df[label] = my_df['vol_' + str(m)] - my_df['vol_' + str(n)]

    # my_df['vol_diff_6_3'] = my_df['vol_6'] - my_df['vol_3']
    # my_df['vol_diff_3_1'] = my_df['vol_3'] - my_df['vol_1']
    # my_df['vol_diff_6_1'] = my_df['vol_6'] - my_df['vol_1']

    for num_months in PRICE_TRENDS:
        my_df['price_trend_' + str(num_months)] = calc_trend_diff(my_df['price'], num_months)
    # my_df['price_trend_2'] = calc_trend_diff(my_df['price'], 2)
    # my_df['price_trend_3'] = calc_trend_diff(my_df['price'], 3)
    # my_df['price_trend_4'] = calc_trend_diff(my_df['price'], 4)
    # my_df['price_trend_5'] = calc_trend_diff(my_df['price'], 5)
    # my_df['price_trend_6'] = calc_trend_diff(my_df['price'], 6)
    for time_period in GROWTH_DICT:
        my_df[time_period] = calc_downstream_returns(my_df, GROWTH_DICT[time_period])
    return my_df
