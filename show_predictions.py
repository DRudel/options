import pickle
import pandas as pd
from datetime import datetime
from fund import Fund, FundModel
from result_evaluator import SingleTanhResultEvaluator
from yahoo_fin.stock_info import get_data, get_live_price
import numpy as np
from large_change_detector import LargeChangeDetector
from get_macro_data import update_macro_data

update_macro_data()
today = datetime.today()

day_of_month = today.day
offset = 17 - day_of_month

#SPY_data = get_data("SPY", start_date=yesterday_string, end_date=tomorrow_string, index_as_date=True, interval="1d")
current_SPY_price = get_live_price('SPY')

SP500_data: pd.DataFrame = pickle.load(open('sp500_daily.pickle', 'rb'))
SP500_fund: Fund = pickle.load(open('sp_v91_slim.pickle', 'rb'))
SP500_results = SP500_fund.predict(SP500_data, price_today=current_SPY_price, num_days_offset=offset)
SP500_results[1].to_csv('SP500_current_predictions.csv', index=False)
SP500_results[0].to_csv('SP500_full_results.csv', index=False)

lc_detector_predictions = []
for num_days in [15, 30, 45]:
    for threshold in [2, 3, 4, 5]:
        if threshold > np.sqrt(num_days):
            continue
        print(f'generating lc predictions for SP at {num_days} days out with {threshold} pct drop.')
        this_detector: LargeChangeDetector = pickle.load(open('lc_detectors/' + str(num_days) + '_' + str(threshold) +
                                                              '_trained.pickle', 'rb'))
        stub = SP500_data.iloc[-2000:]
        results = this_detector.predict(stub)
        final_results = results.iloc[-1]
        this_series = pd.Series([num_days, threshold, this_detector.labels.mean(), final_results['danger']],
                                index=['num_days', 'pct_loss', 'historical', 'danger'])
        lc_detector_predictions.append(this_series)

lc_detector_df = pd.concat(lc_detector_predictions, axis=1).T
lc_detector_df.to_csv('sp_put_dangers.csv', index=False)
print()

current_QQQ_price = get_live_price('QQQ')

N100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
N100_fund: Fund = pickle.load(open('n100_v91_trained.pickle', 'rb'))
N100_results = N100_fund.predict(N100_data, price_today=current_QQQ_price, num_days_offset=offset)
N100_results[1].to_csv('N100_current_predictions.csv', index=False)
N100_results[0].to_csv('N100_full_results.csv', index=False)

#QQQ_data = get_data("QQQ", start_date="01/01/2023", end_date=tomorrow_string, index_as_date=True, interval="1d")
#current_QQQ_price = QQQ_data['close'].iloc[-1]

# n100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
# n100_fund: Fund = pickle.load(open('temp_trained.pickle', 'rb'))
# n100_results = n100_fund.predict(n100_data, price_today=270.54, num_days_offset=offset)
# n100_results[1].to_csv('n100_results.csv')
# print()


# my_evaluator = SingleTanhResultEvaluator(16, 2.5, 0.2)
# # my_fund.tune(my_evaluator)
#
# sp_file = 'SP500_dec_27'
# n100_file = 'N100_dec_27'
#
# SPY_price = 382.03
# QQQ_price = 264.14
# offset = -11

# n100_data = pd.read_csv(n100_file + '.csv')
# n100_fund: Fund = pickle.load(open('N100_v8_trained.pickle', 'rb'))
# n100_results = n100_fund.predict(n100_data, price_today=QQQ_price, num_days_offset=offset)
# n100_results[1].to_csv(n100_file + '_results.csv')
#
# sp_data = pd.read_csv(sp_file + '.csv')
# sp_fund: Fund = pickle.load(open('SP500v8_trained.pickle', 'rb'))
# # sp_fund.train_classifiers(jitter_magnitude=0.15, jitter_count=10)
# # sp_fund.train_regressors(jitter_magnitude=0.15, jitter_count=10)
# sp_results = sp_fund.predict(sp_data, price_today=SPY_price, num_days_offset=offset)
# sp_results[1].to_csv(sp_file + '_results.csv')



