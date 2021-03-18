import multiprocessing

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sources.decision_tree.ensemble_method import EnsembleMethod


class GenerateDecisionTree:
    def __init__(self, ensemble_method, n_estimators, max_depth):
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Constant Parameters
        # Configuration
        self.debug_mode = True
        self.cherry_pick_iterations = 256
        self.num_cores = 14

        # Tree related
        self.ccp_alpha = 0.0001
        self.min_samples_leaf = 2
        self.criterion = "entropy"

        # Execute all needed functions to generate the modell
        self.result = self.cherry_pick()

        if self.debug_mode:
            self.print_evaluation()

    def cherry_pick(self):
        # TODO
        X_train = 0
        Y_train = 0
        X_test_and_opt = 0

        pool = multiprocessing.Pool(processes=self.num_cores)
        args = []
        for i in range(self.cherry_pick_iterations):
            args.append([self.model(i), X_train, Y_train, X_test_and_opt])
        classifier = pool.map(self.evaluate_classifier, args)
        classifier.sort(key=lambda x: x[1], reverse=True)
        return classifier[0][0]

    def max_depth_forest(self):
        return max(x.tree_.max_depth for x in self.result.estimators_)

    def evaluate_classifier(self, args):
        # TODO
        y_test_and_opt = 0

        clf, X_train, y_train, X_test_and_opt = args

        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test_and_opt)

        correct = 0
        for i in range(len(y_test_and_opt)):
            if predicted[i] == y_test_and_opt[i]:
                correct += 1

        accuracy = correct / len(y_test_and_opt)

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

    def save_model_to_file(self, path):
        raise Exception("Not implemented.")

    def print_evaluation(self):
        raise Exception("Not implemented.")
