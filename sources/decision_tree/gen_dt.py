import multiprocessing
import os
import math

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sources.decision_tree.ensemble_method import EnsembleMethod


# TODO: API Kommentare
class GenerateDecisionTree:
    def __init__(self, ensemble_method, n_estimators, max_depth):
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Set random seeds
        np.random.seed(0)
        os.environ['PYTHONHASHSEED'] = str(0)

        # Constant Parameters
        # Configuration
        self.debug_mode = False
        self.cherry_pick_iterations = 128
        self.num_cores = multiprocessing.cpu_count()

        # Tree related
        self.ccp_alpha = 0.0001
        self.min_samples_leaf = 2
        self.criterion = "entropy"

    def fit(self, training_data_x, training_data_y, fraction_opt):
        self.training_data_train_x, self.training_data_opt_x, self.training_data_train_y, \
        self.training_data_opt_y = train_test_split(training_data_x, training_data_y,
                                                    test_size=fraction_opt, random_state=0)
        self.result = self.cherry_pick()

    def cherry_pick(self):
        pool = multiprocessing.Pool(processes=self.num_cores)
        best_classifier = None
        for j in range(int(math.ceil(self.cherry_pick_iterations / self.num_cores))):
            args = []
            for i in range(self.num_cores):
                args.append(self.model(i))
            classifier = pool.map(self.evaluate_classifier, args)
            classifier.sort(key=lambda x: x[1], reverse=True)
            if best_classifier is None or best_classifier[1] < classifier[0][1]:
                best_classifier = classifier[0]
        return best_classifier[0]

    def evaluate_classifier(self, clf):
        clf = clf.fit(self.training_data_train_x, self.training_data_train_y)
        predicted = clf.predict(self.training_data_opt_x)

        correct = 0
        for i in range(len(self.training_data_opt_y)):
            if predicted[i] == self.training_data_opt_y[i]:
                correct += 1

        accuracy = correct / len(self.training_data_opt_y)

        return clf, accuracy

    def model(self, random_state):
        if self.ensemble_method == EnsembleMethod.RandomForest:
            return self.model_random_forest(random_state=random_state)

        if self.ensemble_method == EnsembleMethod.Bagging:
            return self.model_bagging(random_state=random_state)

        if self.ensemble_method == EnsembleMethod.Boosting:
            return self.model_ada_boost(random_state=random_state)

        if self.ensemble_method == EnsembleMethod.ExtraTrees:
            return self.model_extra_trees(random_state=random_state)

        raise Exception("Ensemble Method: '{}' is not implemented.".format(self.ensemble_method))

    def model_random_forest(self, random_state):
        return RandomForestClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                      n_estimators=self.n_estimators,
                                      random_state=random_state, n_jobs=1, ccp_alpha=self.ccp_alpha,
                                      min_samples_leaf=self.min_samples_leaf)

    def model_bagging(self, random_state):
        return BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=self.max_depth,
                                                                            criterion=self.criterion,
                                                                            ccp_alpha=self.ccp_alpha,
                                                                            min_samples_leaf=self.min_samples_leaf),
                                 n_estimators=self.n_estimators, random_state=random_state)

    def model_ada_boost(self, random_state):
        return AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=self.max_depth,
                                                                             criterion=self.criterion,
                                                                             ccp_alpha=self.ccp_alpha,
                                                                             min_samples_leaf=self.min_samples_leaf),
                                  n_estimators=self.n_estimators, random_state=random_state, learning_rate=0.2)

    def model_extra_trees(self, random_state):
        return ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=random_state, n_jobs=1,
                                    max_depth=self.max_depth, ccp_alpha=self.ccp_alpha,
                                    min_samples_leaf=self.min_samples_leaf)

    def max_depth_forest(self):
        return max(x.tree_.max_depth for x in self.result.estimators_)

    def predict(self, data):
        return self.result.predict(data)

    def continued_predict(self, data):
        # Assumes that feature 0 and 1 are previous locations
        data_copy = np.asarray(data).copy()
        predictions = []
        data_copy_len = len(data_copy)
        prediction = self.predict([data_copy[0]])[0]
        predictions.append(prediction)
        prev_predicted_location = prediction
        last_distinct_location = 0
        for i in range(1, data_copy_len):
            prediction = self.predict([data_copy[i]])[0]
            if i < data_copy_len - 1:
                predicted_location = prediction
                if predicted_location != prev_predicted_location and prev_predicted_location != last_distinct_location \
                        and prev_predicted_location > 0:
                    last_distinct_location = prev_predicted_location

                data_copy[i + 1][0] = predicted_location
                data_copy[i + 1][1] = last_distinct_location
                prev_predicted_location = predicted_location
            predictions.append(prediction)
        return predictions

    def save_model_to_file(self, path):
        raise Exception("Not implemented.")

    def evaluate_accuracy(self, prediction, reality):
        correct = 0
        for i in range(len(prediction)):
            if prediction[i] == reality[i]:
                correct = correct + 1

        return correct / len(prediction)

    def feature_importances(self):
        return self.result.feature_importances_
