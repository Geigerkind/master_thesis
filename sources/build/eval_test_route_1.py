import numpy as np

from sources.feature.significant_direction_change import FeatureSignificantDirectionChange
from sources.data.test_route_1 import get_test_route_1_labeled_by_xy
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.feature.acceleration_momentum import FeatureAccelerationMomentum
from sources.feature.acceleration_per_second import FeatureAccelerationPerSecond
from sources.feature.discrete_abs_max import FeatureDiscreteAbsMax
from sources.feature.max import FeatureMax
from sources.feature.mean import FeatureMean
from sources.feature.min import FeatureMin
from sources.feature.standard_deviation import FeatureStandardDeviation

np.random.seed(0)
WINDOW_SIZE = 100

print("Reading data...")
data = get_test_route_1_labeled_by_xy(False, 0.3)

print("Processing features...")
data_features = []
data_label = []
for i in range(WINDOW_SIZE + 1, len(data)):
    window = data.iloc[(i - WINDOW_SIZE):i, :]
    if window.iloc[0]["label"] == 0:
        continue

    data_label.append(window.iloc[0]["label"])

    f_acc_per_s = FeatureAccelerationPerSecond(window[["t_stamp", "x_acc", "y_acc", "z_acc"]].values).feature
    f_acc_momentum = FeatureAccelerationMomentum(window[["t_stamp", "x_acc", "y_acc", "z_acc"]].values).feature
    f_significant_direction_change = FeatureSignificantDirectionChange(window[["x_acc", "y_acc", "z_acc"]].values, 0.5).feature

    x_acc_col_list = window["x_acc"].tolist()
    y_acc_col_list = window["y_acc"].tolist()
    z_acc_col_list = window["z_acc"].tolist()
    x_ang_col_list = window["x_ang"].tolist()
    y_ang_col_list = window["y_ang"].tolist()
    z_ang_col_list = window["z_ang"].tolist()

    data_features.append([
        f_significant_direction_change[0],
        f_significant_direction_change[1],
        f_significant_direction_change[2],

        FeatureDiscreteAbsMax(window.iloc[0][["x_acc", "y_acc", "z_acc"]].values).feature,
        FeatureDiscreteAbsMax(window.iloc[0][["x_ang", "y_ang", "z_ang"]].values).feature,

        FeatureMean(x_acc_col_list).feature,
        FeatureMean(y_acc_col_list).feature,
        FeatureMean(z_acc_col_list).feature,
        FeatureMean(x_ang_col_list).feature,
        FeatureMean(y_ang_col_list).feature,
        FeatureMean(z_ang_col_list).feature,

        f_acc_per_s[0],
        f_acc_per_s[1],
        f_acc_per_s[2],

        f_acc_momentum[0],
        f_acc_momentum[1],
        f_acc_momentum[2],

        FeatureMax(x_acc_col_list).feature,
        FeatureMax(y_acc_col_list).feature,
        FeatureMax(z_acc_col_list).feature,
        FeatureMax(x_ang_col_list).feature,
        FeatureMax(y_ang_col_list).feature,
        FeatureMax(z_ang_col_list).feature,

        FeatureMin(x_acc_col_list).feature,
        FeatureMin(y_acc_col_list).feature,
        FeatureMin(z_acc_col_list).feature,
        FeatureMin(x_ang_col_list).feature,
        FeatureMin(y_ang_col_list).feature,
        FeatureMin(z_ang_col_list).feature,

        FeatureStandardDeviation(x_acc_col_list).feature,
        FeatureStandardDeviation(y_acc_col_list).feature,
        FeatureStandardDeviation(z_acc_col_list).feature,
        FeatureStandardDeviation(x_ang_col_list).feature,
        FeatureStandardDeviation(y_ang_col_list).feature,
        FeatureStandardDeviation(z_ang_col_list).feature,
    ])

X_train = data_features[:int(len(data_features) / 2 + 1)]
Y_train = data_label[:int(len(data_label) / 2 + 1)]

X_test = data_features[int(len(data_features) / 2):]
Y_test = data_label[int(len(data_label) / 2):]

print("Training model...")
model = GenerateDecisionTree(EnsembleMethod.RandomForest, 16, 20, X_train, Y_train, 0.5)

prediction = model.predict(X_test)

print("Prediction Accuracy:")
print(model.evaluate_accuracy(prediction, Y_test))

print("Fraction of test data that is 0:")
count = 0
for i in range(len(Y_test)):
    if Y_test[i] == 0:
        count = count + 1
print(count / len(Y_test))
