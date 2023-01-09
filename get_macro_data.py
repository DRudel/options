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
    # cpi_data = pd.DataFrame(
    #     {
    #         'cpi': fred.get_series('CPIAUCSL')
    #     }
    # )
    #
    # prep_cpi = prepare_cpi_data(cpi_data)
    # pickle.dump(prep_cpi, open('cpi_data.pickle', 'wb'))
    #
    # ffund_data = pd.DataFrame(
    #     {
    #         'ff_rate': fred.get_series('FEDFUNDS')
    #     }
    # )
    #
    # prep_ff = prepare_fed_funds_data(ffund_data)
    # pickle.dump(prep_ff, open('ffund_data.pickle', 'wb'))
    #
    #
    # ue_data = pd.DataFrame(
    #     {
    #         'ue_rate': fred.get_series('UNRATE')
    #     }
    # )
    #
    # prep_ue = prep_unemployment_data(ue_data)
    # pickle.dump(prep_ue, open('ue_data.pickle', 'wb'))