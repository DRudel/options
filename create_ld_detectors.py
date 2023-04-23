from get_macro_data import update_macro_data
import pandas as pd
import pickle
from large_change_detector import LargeChangeDetector
from features_v2 import GROWTH_DICT
import numpy as np

# update_macro_data()
sp_data: pd.DataFrame = pickle.load(open('sp500_daily.pickle', 'rb'))

# data: pd.DataFrame = pickle.load(open('n100_daily.pickle', 'rb'))


# look_ahead_num_days = list(GROWTH_DICT.values())
# look_ahead_num_days = look_ahead_num_days[:1]

for num_days in [15, 30, 45]:
    for threshold in [2, 3, 4, 5]:
        if threshold > np.sqrt(num_days):
            continue
        print(f'creating detector for threshold = {threshold} and number of days = {num_days}')
        this_detector = LargeChangeDetector(str(num_days) + '_' + str(threshold),
                                            base_data=sp_data, pct_change_threshold=threshold, num_days=num_days)
        this_detector.select_features(num_selection_bundles=12, num_trials=5, jitter_count=4, master_seed=9,
                                      cont_jitter_magnitude=0.15, frac=0.2)
        this_detector.save('features_selected')
        this_detector.select_leaf_count(num_selection_bundles=12, num_trials=5, jitter_count=4, master_seed=713,
                                      cont_jitter_magnitude=0.15, frac=0.5)
        this_detector.save('num_leaves_selected')
        this_detector.train(num_selection_bundles=12, num_trials=5, jitter_count=4, master_seed=571,
                                      cont_jitter_magnitude=0.15)
        this_detector.save('trained')
