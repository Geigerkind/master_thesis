from enum import Enum

# Helper enum to specify what kind of DecisionTree ensemble should be trained in the GenerateDecisionTree class
class EnsembleMethod(Enum):
    RandomForest = 1
    Bagging = 2
    Boosting = 3
    ExtraTrees = 4
