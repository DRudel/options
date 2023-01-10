import pickle
import pandas as pd
from datetime import datetime
from fund import Fund, FundModel
from result_evaluator import SingleTanhResultEvaluator
from yahoo_fin.stock_info import get_data
import numpy as np

today = datetime.today()
tomorrow = pd.to_datetime(today) + np.timedelta64(1, 'D')
tomorrow_string = tomorrow.date()

day_of_month = today.day
offset = 17 - day_of_month

QQQ_data = get_data("QQQ", start_date="01/01/2023", end_date=tomorrow_string, index_as_date=True, interval="1d")
#current_QQQ_price = QQQ_data['close'].iloc[-1]

n100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
n100_fund: Fund = pickle.load(open('temp_trained.pickle', 'rb'))
n100_results = n100_fund.predict(n100_data, price_today=270.54, num_days_offset=offset)
n100_results[1].to_csv('n100_results.csv')
print()


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



