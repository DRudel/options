from features_v2 import prepare_cpi_data, prepare_fed_funds_data, prep_unemployment_data
import pickle
import fredapi as fa

FRED_KEY = 'd93a3b14c829ea815a04422516b3785d'

fred = fa.Fred(api_key=FRED_KEY)

def update_series(series_name, processor, filename):
    raw_data = fred.get_series(series_name)
    processed_data = processor(raw_data)
    pickle.dump(processed_data, open(filename + '.pickle', 'wb'))


def update_macro_data():
    update_series('CPIAUCSL', prepare_cpi_data, 'cpi_data')
    update_series('FEDFUNDS', prepare_fed_funds_data, 'ffund_data')
    update_series('UNRATE', prep_unemployment_data, 'ue_data')
