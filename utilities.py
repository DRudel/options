from pricing import euro_vanilla


def calc_final_value(change, threshold):
    return_values = change.copy() - threshold
    return_values[return_values < 0] = 0
    return return_values


def calculate_prices(row, margin, vol_name, num_months, interest_rate=0.015, vol_factor=1, base_factor=0):
    spot = row['price']
    strike = spot * (1 + margin/100)
    volatility = base_factor + vol_factor * row[vol_name] / 100
    time = num_months / 12
    value = euro_vanilla(spot, strike, time, interest_rate, volatility)
    return value / spot
