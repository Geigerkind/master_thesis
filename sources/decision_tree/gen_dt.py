import copy
import math
import multiprocessing
import os

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.config import NUM_CORES



class GenerateDecisionTree:
    def __init__(self, ensemble_method, n_estimators, max_depth):
        """
        :param ensemble_method: What kind of Ensemble-Method found in the enum "EnsembleMethod" should be used
        :param n_estimators: How many decision trees should be used in the ensemble
        :param max_depth: The maximum depth of a decision tree in the ensemble
        """

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
        self.num_cores = NUM_CORES

        # Tree related
        self.ccp_alpha = 0.0001
        self.min_samples_leaf = 2
        self.criterion = "entropy"

    def fit(self, training_data_x, training_data_y, fraction_opt):
        """
        We fit a little bit different as usual.
        How its exactly done, please refer to the training chapter in the thesis.

        :param training_data_x: Array of all feature sets that should be used for fitting
        :param training_data_y: Array of all labels to the feature sets
        :param fraction_opt: The fraction of provided sets that should be used for monte carlo optimization
        :return: VOID: You can find the resulting model in self.result
        """
        self.training_data_train_x, self.training_data_opt_x, self.training_data_train_y, \
        self.training_data_opt_y = train_test_split(training_data_x, training_data_y,
                                                    test_size=fraction_opt, random_state=0)
        self.result = self.cherry_pick()

    def cherry_pick(self):
        """
        It evaluates self.cherry_pick_iterations with different random seeds to find the optimal
        decision tree for the specifications.
        Generating a decision tree is not deterministic, hence the monte carlo method.

        :return: Best model from the monte carlo optimization
        """
        best_classifier = None
        num_iterations = int(math.ceil(self.cherry_pick_iterations / self.num_cores))
        with multiprocessing.get_context("spawn").Pool(processes=self.num_cores) as pool:
            for j in range(num_iterations):
                args = []
                for i in range(self.num_cores):
                    args.append(self.model(i + j * self.num_cores))
                classifier = pool.map(self.evaluate_classifier, args)
                classifier.sort(key=lambda x: x[1], reverse=True)
                if best_classifier is None or best_classifier[1] < classifier[0][1]:
                    best_classifier = classifier[0]
        return best_classifier[0]

    def evaluate_classifier(self, clf):
        """
        Helper method for cherry_pick.
        Evaluates a classifier based on the fraction_opt provided in fit.

        :param clf: model to be evaluated
        :return: model and its accuracy on the optimization test set
        """

        clf = clf.fit(self.training_data_train_x, self.training_data_train_y)
        predicted = clf.predict(self.training_data_opt_x)

        correct = 0
        for i in range(len(self.training_data_opt_y)):
            if predicted[i] == self.training_data_opt_y[i]:
                correct += 1

        accuracy = correct / len(self.training_data_opt_y)

        return clf, accuracy

    def model(self, random_state):
        """
        Helper method to get the correct Ensemble Model from Scikit-Learn.
        :param random_state: An integer number
        :return: Model Template
        """

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
        """
        Ensemble Method Template: RandomForest
        """
        return RandomForestClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                      n_estimators=self.n_estimators,
                                      random_state=random_state, n_jobs=1, ccp_alpha=self.ccp_alpha,
                                      min_samples_leaf=self.min_samples_leaf)

    def model_bagging(self, random_state):
        """
        Ensemble Method Template: Bagging
        """
        return BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=self.max_depth,
                                                                            criterion=self.criterion,
                                                                            ccp_alpha=self.ccp_alpha,
                                                                            min_samples_leaf=self.min_samples_leaf),
                                 n_estimators=self.n_estimators, random_state=random_state)

    def model_ada_boost(self, random_state):
        """
        Ensemble Method Template: AdaBoost
        """
        return AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=self.max_depth,
                                                                             criterion=self.criterion,
                                                                             ccp_alpha=self.ccp_alpha,
                                                                             min_samples_leaf=self.min_samples_leaf),
                                  n_estimators=self.n_estimators, random_state=random_state, learning_rate=0.2)

    def model_extra_trees(self, random_state):
        """
        Ensemble Method Template: ExtraTrees
        """
        return ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=random_state, n_jobs=1,
                                    max_depth=self.max_depth, ccp_alpha=self.ccp_alpha,
                                    min_samples_leaf=self.min_samples_leaf)

    def max_depth_forest(self):
        """
        We specify a max-depth but it may not be fully used.
        This method gives the maximum depth of a tree found in the ensemble.

        :return: maximum depth as an integer
        """
        return max(x.tree_.max_depth for x in self.result.estimators_)

    def predict(self, data):
        """
        Uses the predict function of the model.

        :param data: Array of feature sets that should be predicted
        :return: Array of discrete results
        """
        return self.result.predict(data)

    def predict_proba(self, data):
        """
        Uses ther predict_proba function of the model.

        :param data: Array of feature sets that should be predicted
        :return: Array of results encoded as CDFs
        """
        return self.result.predict_proba(data)

    def continued_predict_proba(self, data):
        """
        Predict the results of the provided test data iteratively by using the predict function.
        Instead of using the provided previous location, it uses its predicted previous location.
        Hence the data series must be in order.

        :param data: Array of feature sets that should be predicted
        :return: Array of discrete results
        """
        return self.__continued_predict(data, True)

    def continued_predict(self, data):
        """
        Predict the results of the provided test data iteratively by using the predict_proba function.
        Instead of using the provided previous location, it uses its predicted previous location.
        Hence the data series must be in order.

        :param data: Array of feature sets that should be predicted
        :return: Array of results encoded as CDFs
        """
        return self.__continued_predict(data, False)

    def __continued_predict(self, data, use_predict_proba=False):
        """
        See continued_predict.
        """
        # Assumes that feature 0 and 1 are previous locations
        data_copy = np.asarray(data).copy()
        predictions = []
        data_copy_len = len(data_copy)
        data_copy[0][0] = 0
        data_copy[0][1] = 0
        prediction = self.predict([data_copy[0]])[0]
        if use_predict_proba:
            predictions.append(self.predict_proba([data_copy[0]])[0])
        else:
            predictions.append(prediction)
        last_distinct_locations = [0, 0]
        for i in range(1, data_copy_len):
            prediction_proba = self.predict_proba(data_copy[i:i+1])[0]
            prediction = np.asarray(prediction_proba).argmax()
            if i < data_copy_len - 1:
                predicted_location = prediction
                if 0 < predicted_location != last_distinct_locations[-1]:
                    last_distinct_locations.append(predicted_location)
                    last_distinct_locations.pop(0)

                data_copy[i + 1][0] = predicted_location
                if predicted_location == 0:
                    data_copy[i + 1][1] = last_distinct_locations[-1]
                else:
                    data_copy[i + 1][1] = last_distinct_locations[-2]

            if use_predict_proba:
                predictions.append(prediction_proba)
            else:
                predictions.append(prediction)
        return predictions

    def evaluate_accuracy(self, prediction, reality):
        """
        Compares predicted data and actual data.

        :param prediction: Array of predictions
        :param reality: Array of actual labels
        :return: Accuracy (float)
        """
        correct = 0
        for i in range(len(prediction)):
            if prediction[i] == reality[i]:
                correct = correct + 1

        return correct / len(prediction)

    def feature_importances(self):
        """
        :return: self.result.feature_importances_
        """
        return self.result.feature_importances_

    def permutation_importance(self, test_features, test_labels):
        """
        This was proposed by Leo Breiman in the Random Forest paper.
        It calculates the accuracy for a provided data set.
        We then shuffle a feature and see the error we get compared to the correct order.
        The higher the error, the more important the feature is.

        :param fitted_model: The fitted model
        :param test_features: The feature sets to infer the importance on
        :param test_labels: Labels for the feature sets
        :return: Array of errors for each feature, higher is more important
        """

        test_predictions = self.predict(test_features)
        test_accuracy = self.evaluate_accuracy(test_predictions, test_labels)

        importances = []
        test_len = len(test_features)
        for i in range(len(test_features[0])):
            # Shuffle column i
            permutation = np.random.permutation(test_len)
            copy_test_features = copy.deepcopy(test_features)
            for ctf_index in range(test_len):
                copy_test_features[ctf_index][i] = test_features[permutation[ctf_index]][i]

            # Calculate accuracy
            ctf_predictions = self.predict(copy_test_features)
            ctf_accuracy = self.evaluate_accuracy(ctf_predictions, test_labels)

            importances.append(abs(test_accuracy - ctf_accuracy))

        return importances
