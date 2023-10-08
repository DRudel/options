from features_v2 import prepare_cpi_data, prepare_fed_funds_data, prep_unemployment_data, prep_t10_data, \
    prep_t10_1_data, prep_t1_data, prep_t10_fed_data, prep_t1_fed_data
import pickle
import fredapi as fa

FRED_KEY = 'd93a3b14c829ea815a04422516b3785d'

fred = fa.Fred(api_key=FRED_KEY)

def update_series(series_name, processor, filename, provided=None):
    if provided is None:
        raw_data = fred.get_series(series_name)
    else:
        raw_data = provided
        raw_data = raw_data.dropna()
    processed_data = processor(raw_data)
    pickle.dump(processed_data, open(filename + '.pickle', 'wb'))


def update_macro_data():
    t1_yields = fred.get_series('DGS1')
    t10_yields = fred.get_series('DGS10')
    yield_diff = t10_yields.subtract(t1_yields).rolling('5D').mean()
    update_series('CPIAUCSL', prepare_cpi_data, 'cpi_data')
    update_series('FEDFUNDS', prepare_fed_funds_data, 'ffund_data')
    update_series('UNRATE', prep_unemployment_data, 'ue_data')
    update_series('DGS10', prep_t10_data, 't10_data')
    update_series(None, prep_t10_1_data, 't10_1_data', provided=yield_diff)
    update_series('DGS1', prep_t1_data, 't1_data')
    update_series('T10YFF', prep_t10_fed_data, 't10_ffund_data')
    update_series('T1YFF', prep_t1_fed_data, 't1_ffund_data')
