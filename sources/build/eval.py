import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from sources.config import BIN_FOLDER_PATH
from sources.data.data_compiler import DataCompiler
from sources.data.data_set import DataSet
from sources.data.features import Features
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.ffnn.gen_ffnn import GenerateFFNN

"""
This file uses the Data Compiler in order to train the decision tree and FFNN 
just like how it is described in the thesis.
"""

_, encode_paths_between_as_location, dt_forest_size, dt_max_height, ffnn_num_hidden_layers, \
ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs, input_data_sets = sys.argv
input_data_sets = input_data_sets.split(',')
input_data_sets = [int(x) for x in input_data_sets]
res_input_data_sets = []
if 1 in input_data_sets:
    res_input_data_sets.append(DataSet.SimpleSquare)
if 2 in input_data_sets:
    res_input_data_sets.append(DataSet.LongRectangle)
if 3 in input_data_sets:
    res_input_data_sets.append(DataSet.RectangleWithRamp)
if 4 in input_data_sets:
    res_input_data_sets.append(DataSet.ManyCorners)

evaluation_name = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(encode_paths_between_as_location, dt_forest_size,
                                                                      dt_max_height, ffnn_num_hidden_layers,
                                                                      ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs,
                                                                      "".join([str(x) for x in input_data_sets]))

encode_paths_between_as_location = 1 == int(encode_paths_between_as_location)
dt_forest_size = int(dt_forest_size)
dt_max_height = int(dt_max_height)
ffnn_num_hidden_layers = int(ffnn_num_hidden_layers)
ffnn_num_nodes_per_hidden_layer = int(ffnn_num_nodes_per_hidden_layer)
ffnn_num_epochs = int(ffnn_num_epochs)

np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

FRACTION_PREDICTION_LABELED = 0.5
NUM_EPOCHS_PER_CYCLE = ffnn_num_epochs

features = [Features.PreviousLocation, Features.AccessPointDetection, Features.Temperature,
            Features.Heading, Features.Volume, Features.Time, Features.Angle, Features.Acceleration, Features.Light]
data = DataCompiler(res_input_data_sets, features, False, encode_paths_between_as_location)
# data = DataCompiler(res_input_data_sets, features, True, encode_paths_between_as_location)

print("Saving data...")
try:
    os.mkdir(BIN_FOLDER_PATH + "/" + evaluation_name + "/")
except:
    pass

with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_data.pkl", 'wb') as file:
    pickle.dump(data, file)

print("")
print("Training models:")
print("Training...")
acc_per_cycle = []

print("Creating validation set...")
dt_vs_features = []
dt_vs_labels = []
knn_vs_features = []
knn_vs_labels = []
for cycle in range(data.num_cycles - data.num_validation_cycles, data.num_cycles):
    for data_set_index in range(len(data.result_features_dt)):
        dt_vs_features = dt_vs_features + data.result_features_dt[data_set_index][cycle]
        dt_vs_labels = dt_vs_labels + data.result_labels_dt[data_set_index][cycle]
        knn_vs_features = knn_vs_features + data.result_features_knn[data_set_index][cycle]
        knn_vs_labels = knn_vs_labels + data.result_labels_knn[data_set_index][cycle]

dt_data_features = []
dt_data_labels = []
knn_data_features = []
knn_data_labels = []

dt_next_cycle_features = []
knn_next_cycle_features = []
dt_next_cycle_labels = []
knn_next_cycle_labels = []
# BIG NOTE: We change the data that is present in "data", because this is a reference not a copy
for data_set_index in range(len(data.result_features_dt)):
    dt_next_cycle_features = dt_next_cycle_features + data.result_features_dt[data_set_index][0]
    dt_next_cycle_labels = dt_next_cycle_labels + data.result_labels_dt[data_set_index][0]
    knn_next_cycle_features = knn_next_cycle_features + data.result_features_knn[data_set_index][0]
    knn_next_cycle_labels = knn_next_cycle_labels + data.result_labels_knn[data_set_index][0]

log_acc_per_cycle = open(BIN_FOLDER_PATH + "/" + evaluation_name + "/log_accuracy_per_cycle.csv", "w")
log_acc_per_cycle.write("cycle,accuracy_dt,accuracy_knn\n")

log_knn_hist = open(BIN_FOLDER_PATH + "/" + evaluation_name + "/log_knn_history.csv", "w")
log_knn_hist.write("cycle,index,loss,accuracy,val_loss,val_accuracy\n")

model_knn = 0
model_dt = 0
for cycle in range(data.num_cycles - data.num_validation_cycles):
    print("Training cycle: {0}".format(cycle))
    print("Initializing...")
    # Reinitializing to make sure that there is no partial learning
    model_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, dt_forest_size, dt_max_height)
    model_knn = GenerateFFNN(data.num_inputs, data.num_outputs, ffnn_num_hidden_layers, ffnn_num_nodes_per_hidden_layer,
                             ffnn_num_epochs)

    print("Preparing input data...")
    dt_data_features = dt_data_features + dt_next_cycle_features
    dt_data_labels = dt_data_labels + dt_next_cycle_labels
    knn_data_features = knn_data_features + knn_next_cycle_features
    knn_data_labels = knn_data_labels + knn_next_cycle_labels

    dt_next_cycle_features = []
    knn_next_cycle_features = []
    dt_next_cycle_labels = []
    knn_next_cycle_labels = []
    for data_set_index in range(len(data.result_features_dt)):
        dt_next_cycle_features = dt_next_cycle_features + data.result_features_dt[data_set_index][cycle + 1]
        dt_next_cycle_labels = dt_next_cycle_labels + data.result_labels_dt[data_set_index][cycle + 1]
        knn_next_cycle_features = knn_next_cycle_features + data.result_features_knn[data_set_index][cycle + 1]
        knn_next_cycle_labels = knn_next_cycle_labels + data.result_labels_knn[data_set_index][cycle + 1]

    print("Training Decision Tree Model...")
    model_dt.fit(dt_data_features, dt_data_labels, 0.25)
    dt_prediction = model_dt.predict(dt_next_cycle_features)
    print("Accuracy: {0}".format(
        model_dt.evaluate_accuracy(dt_prediction, dt_next_cycle_labels)))

    print("")
    print("Training KNN Model...")
    model_knn.fit(knn_data_features, knn_data_labels, knn_vs_features, knn_vs_labels)
    # model_knn.fit(knn_features[cycle], knn_labels[cycle], knn_vs_features, knn_vs_labels)
    knn_prediction = model_knn.predict(knn_next_cycle_features)
    print("Accuracy: {0}".format(
        model_knn.evaluate_accuracy(knn_prediction, knn_next_cycle_labels)))

    if cycle >= data.num_warmup_cycles and cycle < data.num_cycles - data.num_validation_cycles - 1 and Features.PreviousLocation in features:
        print("")
        print("Relabeling next cycle's set...")
        last_distinct_location_dt = dt_next_cycle_features[0][1]
        last_distinct_location_knn = knn_next_cycle_features[0][1]
        permutation = np.random.permutation(len(dt_next_cycle_features))
        frac_pred_labeled = min(1, FRACTION_PREDICTION_LABELED + (1 / 288) * (cycle ** 2))
        for perm_index in range(1, int(len(dt_next_cycle_features) * frac_pred_labeled)):
            i = permutation[perm_index]
            if i == 0:
                continue

            dt_pred = dt_prediction[i]
            dt_prev_pred = dt_prediction[i - 1]

            if dt_pred != dt_prev_pred and dt_prev_pred != last_distinct_location_dt and dt_prev_pred > 0:
                last_distinct_location_dt = dt_prev_pred

            prev_knn_pred = np.array(knn_prediction[i - 1]).argmax() / data.num_outputs
            knn_pred = np.array(knn_prediction[i]).argmax() / data.num_outputs
            if knn_pred != prev_knn_pred and prev_knn_pred != last_distinct_location_knn and prev_knn_pred > 0:
                last_distinct_location_knn = prev_knn_pred

            dt_next_cycle_features[i][0] = dt_prev_pred
            dt_next_cycle_features[i][1] = last_distinct_location_dt
            knn_next_cycle_features[i][0] = prev_knn_pred
            knn_next_cycle_features[i][1] = last_distinct_location_knn

    print("")
    dt_acc = model_dt.evaluate_accuracy(model_dt.predict(dt_vs_features), dt_vs_labels)
    knn_acc = model_knn.evaluate_accuracy(model_knn.predict(knn_vs_features), knn_vs_labels)
    acc_per_cycle.append((dt_acc, knn_acc))

    print("Accuracy on validation set:")
    print("Accuracy DT: {0}".format(dt_acc))
    print("Accuracy KNN: {0}".format(knn_acc))

    print("")

    log_acc_per_cycle.write("{0},{1},{2}\n".format(cycle, dt_acc, knn_acc))
    knn_hist = model_knn.get_history()
    for i in range(NUM_EPOCHS_PER_CYCLE):
        log_knn_hist.write(
            "{0},{1},{2},{3},{4},{5}\n".format(cycle, i, knn_hist["loss"][i], knn_hist["accuracy"][i],
                                               knn_hist["val_loss"][i], knn_hist["val_accuracy"][i]))

log_acc_per_cycle.close()
log_knn_hist.close()

print("Saving models...")
with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'wb') as file:
    pickle.dump(model_dt, file)

model_knn.save(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5")

print("")

print("Collecting data...")
knn_hist = model_knn.get_history()

print("")

print("Generating fancy plots...")
# Accuracy on the validation set over cycles
fig, ax1 = plt.subplots()
ax1.plot(range(data.num_cycles - data.num_validation_cycles), [x[0] for x in acc_per_cycle], "o-g")
ax1.plot(range(data.num_cycles - data.num_validation_cycles), [x[1] for x in acc_per_cycle], "*-b")
ax1.set_xlabel("Zyklus")
ax1.set_ylabel("Klassifizierungsgenauigkeit")
ax1.set_ylim([0, 1])
ax1.set_title("Klassifizierungsgenauigkeit über Trainingszyklen")
fig.legend(['Entscheidungsbaum', 'Künstliches Neuronale Netzwerk'], loc='upper left')
plt.savefig(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_acc_per_cycle.png")
plt.clf()
plt.close(fig)

# KNN: Loss and Accuracy
fig, ax1 = plt.subplots()
ax1.plot(range(NUM_EPOCHS_PER_CYCLE), knn_hist["loss"], "o-g")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.set_title("Loss und Klassifizierungsgenauigkeit über Trainingsepochen")
ax2 = ax1.twinx()
ax2.plot(range(NUM_EPOCHS_PER_CYCLE), knn_hist["accuracy"], "*-b")
ax2.set_ylabel("Klassifizierungsgenauigkeit")
ax2.set_ylim([0, 1])
fig.legend(['Loss', 'Klassifizierungsgenauigkeit'], loc='upper left')
plt.savefig(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_loss_acc_training.png")
plt.clf()
plt.close(fig)

# Validation set
fig, ax1 = plt.subplots()
ax1.plot(range(NUM_EPOCHS_PER_CYCLE), knn_hist["val_loss"], "o-g")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.set_title("Validation Loss und Klassifizierungsgenauigkeit über Trainingsepochen")
ax2 = ax1.twinx()
ax2.plot(range(NUM_EPOCHS_PER_CYCLE), knn_hist["val_accuracy"], "*-b")
ax2.set_ylabel("Klassifizierungsgenauigkeit")
ax2.set_ylim([0, 1])
fig.legend(['Loss', 'Klassifizierungsgenauigkeit'], loc='upper left')
plt.savefig(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_loss_acc_validation.png")
plt.clf()
plt.close(fig)
