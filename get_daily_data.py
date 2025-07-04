from yahoo_fin.stock_info import get_data
import pandas as pd
import pickle
from datetime import datetime
import numpy as np

today = datetime.today()
tomorrow = pd.to_datetime(today) + np.timedelta64(1, 'D')
tomorrow_string = tomorrow.date()

n100_daily = get_data("^NDX", start_date="01/01/1970", end_date=tomorrow_string, index_as_date=True, interval="1d")
pickle.dump(n100_daily, open('n100_daily.pickle', 'wb'))

sp500_daily = get_data("^GSPC", start_date="01/01/1970", end_date=tomorrow_string, index_as_date=True, interval="1d")
pickle.dump(sp500_daily, open('sp500_daily.pickle', 'wb'))