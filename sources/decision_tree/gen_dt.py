import multiprocessing

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sources.decision_tree.ensemble_method import EnsembleMethod


class GenerateDecisionTree:
    def __init__(self, ensemble_method, n_estimators, max_depth, training_data_x, training_data_y, fraction_opt):
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y
        self.fraction_opt = fraction_opt

        # Constant Parameters
        # Configuration
        self.debug_mode = False
        self.cherry_pick_iterations = 256
        self.num_cores = 14

        # Tree related
        self.ccp_alpha = 0.0001
        self.min_samples_leaf = 2
        self.criterion = "entropy"

        # Execute all needed functions to generate the model
        self.training_data_train_x, self.training_data_opt_x, self.training_data_train_y, \
        self.training_data_opt_y = train_test_split(self.training_data_x, self.training_data_y,
                                                    test_size=self.fraction_opt, random_state=0)
        self.result = self.cherry_pick()

        if self.debug_mode:
            self.print_evaluation()

    def cherry_pick(self):
        pool = multiprocessing.Pool(processes=self.num_cores)
        args = []
        for i in range(self.cherry_pick_iterations):
            args.append(self.model(i))
        classifier = pool.map(self.evaluate_classifier, args)
        classifier.sort(key=lambda x: x[1], reverse=True)
        return classifier[0][0]

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

    def save_model_to_file(self, path):
        raise Exception("Not implemented.")

    def print_evaluation(self):
        raise Exception("Not implemented.")
