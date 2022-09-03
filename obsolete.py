

def top_two_ranker(df: pd.DataFrame):
    ranked = df.sort_values('score', ascending=False)
    ranked: pd.DataFrame = ranked.iloc[:2].copy()
    return ranked.groupby('test_index', as_index=False).mean()

class DoubleFeatureEvaluationRound:

    def __init__(self, model_prototype, data, established_indexes, possible_indexes, num_bundles,
                 results_evaluator: ResultEvaluator, classification=True, max_rows_better=8, max_improvement=0.08,
                 pairs_to_evaluate=None, test_singletons=True, min_features=2, **kwargs):
        self.model = clone(model_prototype)
        self.established_indexes = established_indexes
        self.possible_indexes = possible_indexes
        self.evaluator = results_evaluator
        self.trial_results = None
        self.classification = classification
        self.pairs_to_evaluate = pairs_to_evaluate
        self.test_singletons = test_singletons
        self.summary = None
        self.bundles: list[TrainTestBundle] = []
        self.min_features = min_features # if there are fewer than this many features, selection continues
        self.max_rows_better = max_rows_better # If more than this many rows are better, selection continues
        self.max_improvement = max_improvement # If relative improvement is greater than this, selection continues
        for k in range(num_bundles):
            this_bundle = TrainTestBundle(data=data, **kwargs)
            this_bundle.form_trials()
            self.bundles.append(this_bundle)

    def get_test_pairs(self):
        remaining = set(self.possible_indexes) - set(self.established_indexes)
        pairs_to_test = list(combinations(remaining, 2))
        if self.pairs_to_evaluate is not None:
            pairs_to_test = pairs_to_test[: self.pairs_to_evaluate]
        if self.test_singletons:
            pairs_to_test.extend([tuple([x, x]) for x in remaining])
        if len(self.established_indexes) > 0:
            pairs_to_test.append((-1, -1))
        return pairs_to_test

    def compile_data(self):
        pairs_to_test = self.get_test_pairs()
        results_by_bundle = []
        for k, bundle in enumerate(self.bundles):
            bundle_results = self.traverse_pairs(pairs_to_test, bundle)
            bundle_results['bundle_index'] = k
            results_by_bundle.append(bundle_results)
        self.trial_results = pd.concat(results_by_bundle, axis=0)

    def traverse_pairs(self, pairs_to_test, bundle):
        index_cohorts = []
        for pair in pairs_to_test:
            feature_indexes = self.established_indexes + list(pair)
            this_cohort = self.process_bundle(bundle=bundle, feature_indexes=feature_indexes, pair=pair)
            index_cohorts.append(this_cohort)
        return pd.concat(index_cohorts, axis=0)

    def process_bundle(self, bundle, **kwargs):
        result_chunks = []
        for trial in bundle.trials:
            this_result = self.run_trial(trial, **kwargs)
            result_chunks.append(this_result)
        full_results = pd.concat(result_chunks, axis=0)
        return full_results

    def run_trial(self, trial: TrainTestTrial, feature_indexes, pair):
        feature_indexes = [x for x in feature_indexes if x > -1]
        training_data = trial.train_X.copy()
        training_data = training_data.iloc[:, feature_indexes]
        has_weight = False
        if trial.train_w is not None:
            has_weight = True
            training_data['weight'] = trial.train_w.copy()
        training_data['label'] = trial.train_y.copy()
        self.train(training_data=training_data, has_weight=has_weight)
        evaluation_features = trial.test_X.iloc[:, feature_indexes]
        actuals = trial.test_y
        if self.classification:
            predictions = self.model.predict_proba(evaluation_features)[:, 1]
            if trial.test_w is not None:
                actuals = actuals * trial.test_w
        else:
            predictions = self.model.create_prices_for_thresholds(evaluation_features)
        results = pd.DataFrame(
            {
                'first_index': pair[0],
                'second_index': pair[1],
                'prediction': predictions,
                'actual': actuals
            }, index=trial.test_X.index
        )
        return results

    def train(self, training_data, has_weight=False):
        training_data = training_data.dropna()
        training_labels = training_data.iloc[:, -1]
        if has_weight:
            training_features = training_data.iloc[:, : -2]
            training_weights = training_data.iloc[:, -2]
            self.model.fit(training_features, training_labels, sample_weight=training_weights)
        else:
            training_features = training_data.iloc[:, : -1]
            self.model.fit(training_features, training_labels)

    def summarize_results(self):
        results_by_index = []
        for k in (list(self.possible_indexes) + [-1]):
            first_df = self.trial_results[self.trial_results['first_index'] == k].copy()
            first_df['partner'] = first_df['second_index'].copy()
            second_df = self.trial_results[self.trial_results['second_index'] == k].copy()
            second_df = second_df[second_df['first_index'] != second_df['second_index']]
            second_df['partner'] = second_df['first_index'].copy()
            indexed_df = pd.concat([first_df, second_df], axis=0)
            indexed_df['idx'] = k
            results_by_index.append(indexed_df)
        tagged_results = pd.concat(results_by_index, axis=0)
        summary = tagged_results.groupby(['idx','partner'], as_index=False,
                                         group_keys=False).apply(self.evaluator.score)
        self.summary = summary

    def report_results(self):
        best_run = self.summary.sort_values('score', ascending=False).iloc[0]
        best_score = best_run['score']
        candidates = list(best_run[['test_index', 'partner_index']].unique())
        if len(candidates) == 1:
            best_index = candidates[0]
        else:
            candidate_runs: pd.DataFrame = self.summary[self.summary['test_index'].isin(candidates)]
            rankings = candidate_runs.groupby('test_index', as_index=False).apply(top_two_ranker)
            best_index = rankings.sort_values('score', ascending=False)['test_index'].iloc[0]
        end_selection = self.determine_selection_termination(best_score)
        return best_index, best_score, end_selection, self.summary.copy()

    def determine_selection_termination(self, best_score):
        if len(self.established_indexes) == 0:
            return False
        sorted_results = self.summary.sort_values('score', ascending=False)
        positions = np.flatnonzero(sorted_results['test_index'] == -1)
        dummy_place = positions[0]
        if dummy_place == 0:
            return True
        if len(self.established_indexes) < self.min_features:
            return False
        if dummy_place > self.max_rows_better:
            return False
        dummy_score = sorted_results.iloc[dummy_place]['score']
        if best_score > (1 + self.max_improvement) * dummy_score:
            return False
        return True

class TrainTestTrial:

    def __init__(self, data_sets: list[pd.DataFrame], exclusion_indexes, evaluation_indexes,
                 has_weights=False):
        training_sets = []
        evaluation_sets = []
        for ds in data_sets:
            training_data = select_by_integer_index(ds, exclusion_indexes, False)
            evaluation_data = select_by_integer_index(ds, evaluation_indexes)
            training_sets.append(training_data)
            evaluation_sets.append(evaluation_data)
        full_training_data = pd.concat(training_sets, axis=0)
        full_evaluation_data = pd.concat(evaluation_sets, axis=0)
        self.train_y = full_training_data.iloc[:, -1].copy()
        self.test_y = full_evaluation_data.iloc[:, -1].copy()
        if has_weights:
            self.train_X = full_training_data.iloc[:, : -2].copy()
            self.train_w = full_training_data.iloc[:, -2].copy()
            self.test_w = full_evaluation_data.iloc[:, -2].copy()
            self.test_X = full_evaluation_data.iloc[:, : -2].copy()
        else:
            self.train_X = full_training_data.iloc[:, : -1].copy()
            self.train_w = None
            self.test_w = None
            self.test_X = full_evaluation_data.iloc[:, : -1].copy()


class TrainTestBundle:

    def __init__(self, data, selection_size=None, forward_exclusion_length=7, backward_exclusion_length=10,
                 jitter_count=0, jitter_magnitude=0.2, has_weights=False, min_selection_proportion=0.08,
                 max_selection_proportion=0.2):
        self.data = data.dropna()
        self.labels = self.data.iloc[:, -1].copy()
        if selection_size is None:
            selection_proportion = min_selection_proportion + \
                (max_selection_proportion - min_selection_proportion) * my_rng.random()
            selection_size = int(selection_proportion * (len(data.index) - forward_exclusion_length -
                                                         backward_exclusion_length))
        self.selection_size = selection_size
        self.offset = int(my_rng.random() * (self.selection_size + forward_exclusion_length))
        self.forward_exclusion_length = forward_exclusion_length
        self.backward_exclusion_length = backward_exclusion_length
        self.jitter_count = jitter_count
        self.jitter_magnitude = jitter_magnitude
        self.trials: list[TrainTestTrial] = []
        self.has_weights = has_weights
        self.weights = None
        if self.has_weights:
            self.weights = self.data.iloc[:, -2].copy()

    def form_trials(self):
        if self.weights is None:
            temp_data = self.data.iloc[:, : -1].copy()
        else:
            temp_data = self.data.iloc[:, : -2].copy()
        scaler = StandardScaler()
        temp_data.loc[:, :] = scaler.fit_transform(temp_data)
        jitter_sets = create_jitters(temp_data, self.labels, self.jitter_count, self.jitter_magnitude, self.weights)
        for fold_cutpoint in range(0, int(len(self.data.index - self.offset) / self.selection_size)):
            cut_index = self.offset + self.selection_size * fold_cutpoint
            before_cut = max(cut_index - self.backward_exclusion_length, 0)
            exclusion_slice = slice(before_cut, cut_index + self.selection_size + self.forward_exclusion_length)
            evaluation_slice = slice(cut_index, cut_index + self.selection_size)
            this_trial = TrainTestTrial(jitter_sets, exclusion_slice, evaluation_slice, has_weights=self.has_weights)
            self.trials.append(this_trial)


def create_jitters(features, labels, jitter_count, jitter_magnitude, weights=None):
    labels = labels.copy()
    features = features.copy()
    base_data = features.copy()
    if weights is not None:
        base_data['weight'] = weights
    base_data['label'] = labels
    jitter_sets = [base_data]
    for k in range(jitter_count):
        this_random = 2 * jitter_magnitude * my_rng.random(features.shape)
        this_random = this_random - jitter_magnitude
        this_jitter = features + this_random
        if weights is not None:
            this_jitter['weight'] = weights
        this_jitter['label'] = labels
        jitter_sets.append(this_jitter)
    return jitter_sets