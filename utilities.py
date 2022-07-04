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


def evaluate_factor(factors, df, margin, num_months, vol_name, change_name, penalize_bias=True):
    clean_data = df.copy().dropna(subset=[vol_name, change_name])
    base_factor = factors[1]
    prices = 100 * clean_data.apply(calculate_prices, axis=1, margin=margin, num_months=num_months,
                                    vol_name=vol_name, vol_factor=factors[0], base_factor=base_factor)
    values = calc_final_value(clean_data[change_name], margin)
    diff = prices - values
    squared_error = diff * diff
    if penalize_bias:
        return abs(diff.mean()) + squared_error.mean()
    print(f'mse = {squared_error.mean()}; bias = {diff.mean()}')
    return squared_error.mean(), diff.mean()