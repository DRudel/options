import pandas as pd
from pricing_models import NormPricer
import pickle
import numpy as np

LEFT_TRIM = 150

# sp_data: pd.DataFrame = pickle.load(open('sp500_daily.pickle', 'rb'))
# sp_data = sp_data[['close']]

n100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
n100_data = n100_data[['close']].copy()

def generate_vol_data(input_df, vol_granularity, alpha):
    data = input_df.copy()
    vol_gran_code = str(vol_granularity) + 'd'
    data['abs_interval_change'] = data.rolling(vol_gran_code, closed='both').agg(lambda x: abs(x.iloc[-1] - x.iloc[0]))
    data = data.iloc[vol_granularity:]
    data['rel_interval_change'] = data['abs_interval_change'] / data['close']
    data['square_interval_change'] = np.power(data['rel_interval_change'], 2)
    data['decayed_ms_change'] = data['square_interval_change'].ewm(alpha=alpha).mean()
    data['evol'] = np.sqrt(data['decayed_ms_change'])
    data = data.iloc[LEFT_TRIM - vol_granularity:]
    return data

def generate_growth_data(vol_data, stride, start, stop):
    my_frames = []
    for j in range(start, stop):
        days = stride * j
        day_code = str(days) + 'd'
        min_periods = int(4.5 * days / 7)
        results_df = pd.DataFrame(
            {
                'time': days - 1,
                'order': range(len(vol_data['close'])),
                'price': vol_data['close'],
                'vol': vol_data['evol'],
                'abs_growth': vol_data['close'][::-1].rolling(day_code, closed='both',
                                                           min_periods=min_periods).agg(lambda x: x.iloc[0] - x.iloc[-1])[::-1]
            }
        )
        results_df['growth'] = results_df['abs_growth'] / results_df['price']
        my_frames.append(results_df)
    data = pd.concat(my_frames, axis=0)
    data = data.dropna()
    return data

thresholds = list(range(3, 15))
thresholds = [t/(365 * 25) for t in thresholds]

lowest_loss = 1

for vol_gran in range(3, 30, 2):
    print()
    print("---------------")
    print(f'vol_gran = {vol_gran}')
    for alpha in [0.035]:
        print()
        print(f'alpha = {alpha}')
        my_pricing_model = NormPricer(lower_z_bound=0, upper_z_bound=2.5, num_partitions=100, call=True, decay=0.00005)
        my_pricing_model.vol_name = 'vol'
        vol_data = generate_vol_data(n100_data, vol_granularity=vol_gran, alpha=alpha)
        growth_data = generate_growth_data(vol_data, 10, 4, 20)
        growth_data = growth_data[(growth_data['order'] % 5) == 0]
        # growth_data = growth_data.sample(frac=0.1, random_state=117)
        loss = my_pricing_model.train(growth_data, thresholds=thresholds, var_partitions=4, return_loss=True)
        if loss < lowest_loss:
            lowest_loss = loss

print(f'lowest loss = {lowest_loss}')
