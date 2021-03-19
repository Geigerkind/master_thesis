from sklearn.datasets import make_classification

from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

model = GenerateDecisionTree(EnsembleMethod.RandomForest, 16, 20, X, y, 0.5)
result = model.predict([[0, 0, 0, 0]])

print(result)
