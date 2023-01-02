import pandas as pd
from pricing_models import NormPricer
import pickle
import numpy as np

LEFT_TRIM = 150

sp_data: pd.DataFrame = pickle.load(open('sp500_daily.pickle', 'rb'))
sp_data = sp_data[['close']]

# n100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
# n100_data = n100_data[['close']].copy()

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

thresholds = list(range(3, 30))
thresholds = [t/(365 * 50) for t in thresholds]
my_pricing_model = NormPricer(lower_z_bound=0, upper_z_bound=3, num_partitions=300, call=True, decay=0.00015)
my_pricing_model.vol_name = 'vol'
vol_data = generate_vol_data(sp_data, vol_granularity=5, alpha=0.02)
growth_data = generate_growth_data(vol_data, 10, 4, 7)
growth_data = growth_data.sample(frac=0.1, random_state=117)
my_pricing_model.train(growth_data, thresholds=thresholds, var_partitions=3)
