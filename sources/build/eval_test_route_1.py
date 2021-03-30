from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from sources.data.test_route_1 import get_test_route_1_labeled_by_xy
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.feature.acceleration_momentum import FeatureAccelerationMomentum
from sources.feature.acceleration_per_second import FeatureAccelerationPerSecond
from sources.feature.discrete_abs_max import FeatureDiscreteAbsMax
from sources.feature.max import FeatureMax
from sources.feature.mean import FeatureMean
from sources.feature.min import FeatureMin
from sources.feature.significant_direction_change import FeatureSignificantDirectionChange
from sources.feature.standard_deviation import FeatureStandardDeviation
from sources.ffnn.gen_ffnn import GenerateFFNN

np.random.seed(0)
WINDOW_SIZE = 100
NUM_CYCLES = 10
FRACTION_PREDICTION_LABELED = 0.6
NUM_OUTPUTS = 9
NUM_CORES = 14
NUM_EPOCHS_PER_CYCLE = 100

print("Reading data...")
data = get_test_route_1_labeled_by_xy(False, 0.15, 1)

print("Processing features...")


def calculate_features(i):
    window = data.iloc[(i - WINDOW_SIZE):i, :]
    f_acc_per_s = FeatureAccelerationPerSecond(window[["t_stamp", "x_acc", "y_acc", "z_acc"]].values).feature
    f_acc_momentum = FeatureAccelerationMomentum(window[["t_stamp", "x_acc", "y_acc", "z_acc"]].values).feature
    f_significant_direction_change = FeatureSignificantDirectionChange(window[["x_acc", "y_acc", "z_acc"]].values,
                                                                       0.5).feature

    x_acc_col_list = window["x_acc"].tolist()
    y_acc_col_list = window["y_acc"].tolist()
    z_acc_col_list = window["z_acc"].tolist()
    x_ang_col_list = window["x_ang"].tolist()
    y_ang_col_list = window["y_ang"].tolist()
    z_ang_col_list = window["z_ang"].tolist()

    # The previous location features are by far the best.
    # FeatureStandardDeviation: 55%
    # FeatureMax: 40%
    # FeatureMin: 38%
    # FeatureAccMomentum: 32%
    # FeatureMean: 27%
    # FeatureSignificantDirectionChange: 27%
    # FeatureAccPerSecond: 19%
    # FeatureDiscreteAbsoluteMax: 17%
    return window.iloc[WINDOW_SIZE - 1]["cycle"], window.iloc[WINDOW_SIZE - 1]["location"], \
           window.iloc[WINDOW_SIZE - 2]["location"], [
               window.iloc[WINDOW_SIZE - 2]["location"],
               # window.iloc[WINDOW_SIZE - 1]["prev_location"],

               FeatureStandardDeviation(x_acc_col_list).feature,
               FeatureStandardDeviation(y_acc_col_list).feature,
               FeatureStandardDeviation(z_acc_col_list).feature,
               FeatureStandardDeviation(x_ang_col_list).feature,
               FeatureStandardDeviation(y_ang_col_list).feature,
               FeatureStandardDeviation(z_ang_col_list).feature,

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

               f_acc_momentum[0],
               f_acc_momentum[1],
               f_acc_momentum[2],

               FeatureMean(x_acc_col_list).feature,
               FeatureMean(y_acc_col_list).feature,
               FeatureMean(z_acc_col_list).feature,
               FeatureMean(x_ang_col_list).feature,
               FeatureMean(y_ang_col_list).feature,
               FeatureMean(z_ang_col_list).feature,

               f_significant_direction_change[0],
               f_significant_direction_change[1],
               f_significant_direction_change[2],

               f_acc_per_s[0],
               f_acc_per_s[1],
               f_acc_per_s[2],

               FeatureDiscreteAbsMax(window.iloc[0][["x_acc", "y_acc", "z_acc"]].values).feature,
               FeatureDiscreteAbsMax(window.iloc[0][["x_ang", "y_ang", "z_ang"]].values).feature,
           ]


features_tmp = []
labels_tmp = []
prev_locations_tmp = []
cycles = []
with Pool(processes=NUM_CORES) as pool:
    result = pool.map(calculate_features, range(WINDOW_SIZE + 1, len(data)))
    for (cycle, label, prev_loc, features) in result:
        cycles.append(cycle)
        labels_tmp.append(label)
        prev_locations_tmp.append(prev_loc)
        features_tmp.append(features)

print("Normalizing KNN data...")
sc = StandardScaler()
knn_features_tmp = sc.fit_transform(features_tmp)

print("Fixing location labels...")
for i in range(len(knn_features_tmp)):
    # Manual scaling between 0 and 1
    knn_features_tmp[i][0] = prev_locations_tmp[i] * (1 / NUM_OUTPUTS)

prev_locations_tmp = 0

print("Onehot encoding KNN data...")
ohe_mapping = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    8: [0, 0, 0, 0, 0, 0, 0, 0, 1],
}

# ohe = OneHotEncoder()
# knn_labels_tmp = ohe.fit_transform([[x] for x in labels_tmp]).toarray()
knn_labels_tmp = [ohe_mapping[x] for x in labels_tmp]

print("Reshaping data...")
dt_features = []
dt_labels = []
knn_features = []
knn_labels = []

current_cycle = 1
cycle_dt_features = []
cycle_dt_labels = []
cycle_knn_features = []
cycle_knn_labels = []
for i in range(len(cycles)):
    if cycles[i] > current_cycle:
        current_cycle = cycles[i]
        dt_features.append(cycle_dt_features)
        dt_labels.append(cycle_dt_labels)
        knn_features.append(cycle_knn_features)
        knn_labels.append(cycle_knn_labels)
        cycle_dt_features = []
        cycle_dt_labels = []
        cycle_knn_features = []
        cycle_knn_labels = []

    cycle_dt_features.append(features_tmp[i])
    cycle_dt_labels.append(labels_tmp[i])
    cycle_knn_features.append(knn_features_tmp[i])
    cycle_knn_labels.append(knn_labels_tmp[i])

dt_features.append(cycle_dt_features)
dt_labels.append(cycle_dt_labels)
knn_features.append(cycle_knn_features)
knn_labels.append(cycle_knn_labels)

cycle_dt_features = 0
cycle_dt_labels = 0
cycle_knn_features = 0
cycle_knn_labels = 0
features_tmp = 0
knn_features_tmp = 0
labels_tmp = 0
knn_labels_tmp = 0
cycles = 0

print("")
print("Training models:")
print("Initializing...")
model_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, 16, 20)
model_knn = GenerateFFNN(len(knn_features[0][0]), NUM_OUTPUTS)

print("Training...")
acc_per_cycle = []
metric_knn_loss = []
metric_knn_acc = []
metric_knn_val_loss = []
metric_knn_val_acc = []

dt_data_features = []
dt_data_labels = []
for cycle in range(NUM_CYCLES - 1):
    print("")
    print("Training cycle: {0}".format(cycle))
    print("Training Decision Tree Model...")
    dt_data_features = dt_data_features + dt_features[cycle]
    dt_data_labels = dt_data_labels + dt_labels[cycle]
    model_dt.fit(dt_data_features, dt_data_labels, 0.5)
    dt_prediction = model_dt.predict(dt_features[cycle + 1])
    print("Accuracy: {0}".format(
        model_dt.evaluate_accuracy(dt_prediction, dt_labels[cycle + 1])))

    print("")
    print("Training KNN Model...")
    model_knn.fit(knn_features[cycle], knn_labels[cycle], knn_features[NUM_CYCLES - 1], knn_labels[NUM_CYCLES - 1])
    knn_prediction = model_knn.predict(knn_features[cycle + 1])
    print("Accuracy: {0}".format(
        model_knn.evaluate_accuracy(knn_prediction, knn_labels[cycle + 1])))

    if cycle < NUM_CYCLES - 2:
        print("")
        print("Relabeling next cycle's set...")
        for i in range(int(len(dt_features[cycle + 1]) * FRACTION_PREDICTION_LABELED)):
            dt_features[cycle + 1][i][0] = dt_prediction[i]
            knn_features[cycle + 1][i][0] = np.array(knn_prediction[i]).argmax() * (1 / NUM_OUTPUTS)

    print("")
    dt_acc = model_dt.evaluate_accuracy(model_dt.predict(dt_features[NUM_CYCLES - 1]), dt_labels[NUM_CYCLES - 1])
    knn_acc = model_knn.evaluate_accuracy(model_knn.predict(knn_features[NUM_CYCLES - 1]), knn_labels[NUM_CYCLES - 1])
    acc_per_cycle.append((dt_acc, knn_acc))

    print("Accuracy on validation set:")
    print("Accuracy DT: {0}".format(dt_acc))
    print("Accuracy KNN: {0}".format(knn_acc))

    print("")
    print("Collecting data...")
    knn_hist = model_knn.get_history()
    metric_knn_loss = metric_knn_loss + knn_hist["loss"]
    metric_knn_acc = metric_knn_acc + knn_hist["accuracy"]
    metric_knn_val_loss = metric_knn_val_loss + knn_hist["val_loss"]
    metric_knn_val_acc = metric_knn_val_acc + knn_hist["val_accuracy"]

print("")
print("Generating fancy plots...")
# Accuracy on the validation set over cycles
fig, ax1 = plt.subplots()
ax1.plot(range(NUM_CYCLES - 1), [x[0] for x in acc_per_cycle], "o-g")
ax1.plot(range(NUM_CYCLES - 1), [x[1] for x in acc_per_cycle], "*-b")
ax1.set_xlabel("Zyklus")
ax1.set_ylabel("Klassifizierungsgenauigkeit")
ax1.set_ylim([0, 1])
ax1.set_title("Klassifizierungsgenauigkeit über Trainingszyklen")
fig.legend(['Entscheidungsbaum', 'Künstliches Neuronale Netzwerk'], loc='upper left')
plt.savefig("/home/shino/Uni/master_thesis/bin/tr1_acc_per_cycle.png")
plt.clf()
plt.close(fig)

# KNN: Loss and Accuracy
fig, ax1 = plt.subplots()
ax1.plot(range((NUM_CYCLES - 1) * NUM_EPOCHS_PER_CYCLE), metric_knn_loss, "o-g")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.set_title("Loss und Klassifizierungsgenauigkeit über Trainingsepochen")
ax2 = ax1.twinx()
ax2.plot(range((NUM_CYCLES - 1) * NUM_EPOCHS_PER_CYCLE), metric_knn_acc, "*-b")
ax2.set_ylabel("Klassifizierungsgenauigkeit")
ax2.set_ylim([0, 1])
fig.legend(['Loss', 'Klassifizierungsgenauigkeit'], loc='upper left')
plt.savefig("/home/shino/Uni/master_thesis/bin/tr1_loss_acc_training.png")
plt.clf()
plt.close(fig)

# Validation set
fig, ax1 = plt.subplots()
ax1.plot(range((NUM_CYCLES - 1) * NUM_EPOCHS_PER_CYCLE), metric_knn_val_loss, "o-g")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.set_title("Validation Loss und Klassifizierungsgenauigkeit über Trainingsepochen")
ax2 = ax1.twinx()
ax2.plot(range((NUM_CYCLES - 1) * NUM_EPOCHS_PER_CYCLE), metric_knn_val_acc, "*-b")
ax2.set_ylabel("Klassifizierungsgenauigkeit")
ax2.set_ylim([0, 1])
fig.legend(['Loss', 'Klassifizierungsgenauigkeit'], loc='upper left')
plt.savefig("/home/shino/Uni/master_thesis/bin/tr1_loss_acc_validation.png")
plt.clf()
plt.close(fig)

"""
print("Fraction of test data that is 0:")
count = 0
for i in range(len(Y_test)):
    if Y_test[i] == 0:
        count = count + 1
print(count / len(Y_test))
"""
