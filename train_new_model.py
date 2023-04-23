import pickle
import pandas as pd
from fund import Fund
from fund_model import FundModel
from result_evaluator import SingleTanhResultEvaluator
from pricing_models import NormPricer
from features_v2 import GROWTH_NAMES
from get_macro_data import update_macro_data

update_macro_data()

#n100_data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))
sp_data: pd.DataFrame = pickle.load(open('sp500_daily.pickle', 'rb'))

thresholds = list(range(3, 30))
thresholds = [t/(365 * 50) for t in thresholds]
my_pricing_model = NormPricer(lower_z_bound=0, upper_z_bound=2.5, num_partitions=100, call=True, decay=0.00004,
                              reg_base_mu=750000)
n100_fund: Fund = Fund('n100_v92', base_data=n100_data)
n100_fund.set_pricing_model(thresholds=thresholds, rough=False, price_model_prototype=my_pricing_model, frac=0.3)
n100_fund.save('pricing_set')

my_evaluator = SingleTanhResultEvaluator(16, 2.5, 0.2)
#my_fund: Fund = pickle.load(open('model_snapshots/sp_v91_pricing_set.pickle', 'rb'))
for num_days in GROWTH_NAMES:
    n100_fund.create_models(num_days=num_days, num_selection_bundles=12, results_evaluator=my_evaluator, num_trials=5,
                          jitter_count=4, master_seed=71, cont_jitter_magnitude=0.15, overwrite=False, frac=0.2)
    n100_fund.save()


#n100_fund: Fund = pickle.load(open('model_snapshots/n100_v9_classifiers_tuned.pickle', 'rb'))

n100_fund.tune_classifiers(evaluator=my_evaluator, frac=0.35, num_trials=6, min_improvement=0.05)
n100_fund.save('classifiers_tuned')

n100_fund.tune_regressors(frac=0.35, num_trials=6, min_leaves=4)
n100_fund.save('regressors_tuned')

n100_fund.train_classifiers(jitter_magnitude=0.15, jitter_count=1, frac=None)
n100_fund.train_regressors(jitter_magnitude=0.15, jitter_count=1, frac=None)

n100_fund.save('trained')


# my_fund: Fund = Fund('sp_v93', base_data=sp_data)
# my_fund.set_pricing_model(thresholds=thresholds, rough=False, price_model_prototype=my_pricing_model, frac=0.3)
# my_fund.save('pricing_set')
# my_fund.report_prices(sp_data, price_today=390.58, num_days_offset=10)


# # my_fund: Fund = pickle.load(open('model_snapshots/temp_pricing_set.pickle', 'rb'))
#
# #my_fund = pickle.load(open('SP500v8_pricing_set.pickle', 'rb'))
#
# # my_fund: Fund = pickle.load(open('SP500_v7_2022-09-08 21_14_23.pickle', 'rb'))
# # pickle.dump(my_fund.pricing_model, open('N100_pricing_model.pickle', 'wb'))
# # pricing_model = pickle.load(open('N100_pricing_model.pickle', 'rb'))
# # my_fund.pricing_model = pricing_model
# # my_fund.set_growth_data()
#
# # for key in my_trained_fund.models:
# #     my_fund.models[key] = my_trained_fund.models[key]
#
# # my_fund: Fund = pickle.load(open('SandP500_v6_2022-07-16 02_45_37.pickle', 'rb'))
#
# # old_fund: Fund= pickle.load(open('N100_v7_2022-09-08 20_11_07.pickle', 'rb'))
# # my_fund: Fund = Fund.extend_fund(old_fund, name='N100_v7_expanded', base_data=my_data)
# # my_fund.pricing_model = my_ref_fund.pricing_model
#
# # my_fund: Fund = pickle.load(open('temp_pricing_set.pickle', 'rb'))
#

# my_fund: Fund = pickle.load(open('model_snapshots/sp_v91_2023-01-14 11_02_38.pickle', 'rb'))

# my_evaluator = SingleTanhResultEvaluator(16, 2.5, 0.2)
# #my_fund: Fund = pickle.load(open('model_snapshots/sp_v91_pricing_set.pickle', 'rb'))
# for num_days in GROWTH_NAMES:
#     my_fund.create_models(num_days=num_days, num_selection_bundles=12, results_evaluator=my_evaluator, num_trials=5,
#                           jitter_count=4, master_seed=71, cont_jitter_magnitude=0.15, overwrite=False, frac=0.2)
#     my_fund.save()

# my_fund: Fund = pickle.load(open('model_snapshots/sp_v91_2023-01-13 06_55_19.pickle', 'rb'))

# my_fund.tune_classifiers(evaluator=my_evaluator, frac=0.35, num_trials=6, min_improvement=0.05)
# my_fund.save('classifiers_tuned')
#
# # my_fund = pickle.load(open('model_snapshots/sp_v90_classifiers_tuned.pickle', 'rb'))
#
# my_fund.tune_regressors(frac=0.35, num_trials=6, min_leaves=4)
# my_fund.save('regressors_tuned')
#
# my_fund.train_classifiers(jitter_magnitude=0.15, jitter_count=1, frac=None)
# my_fund.train_regressors(jitter_magnitude=0.15, jitter_count=1, frac=None)
#
# my_fund.save('trained')
