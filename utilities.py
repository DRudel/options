from pricing import euro_vanilla
from features import GROWTH_DICT
import pandas as pd

def form_pricing_data(data, growth_name, vol_name, include_growths=True):
    this_data = pd.DataFrame({
        'vol': data[vol_name] / 100,
        'time': GROWTH_DICT[growth_name] * (365.25 / 12)
    })
    if include_growths:
        this_data['growth'] = data[growth_name] / 100
    this_data.dropna(inplace=True, axis=0)
    return this_data


def calc_final_value(changes, threshold):
    returns = (changes > threshold) * (changes - threshold)
    # return_values = change.copy() - threshold
    # return_values[return_values < 0] = 0
    return returns


def calculate_prices(row, margin, vol_name, num_months, interest_rate=0.015, vol_factor=1, base_factor=0):
    spot = row['price']
    strike = spot * (1 + margin/100)
    volatility = base_factor + vol_factor * row[vol_name] / 100
    time = num_months / 12
    value = euro_vanilla(spot, strike, time, interest_rate, volatility)
    return value / spot
