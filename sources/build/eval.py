import pickle

import matplotlib.pyplot as plt
import numpy as np

from sources.data.data_compiler import DataCompiler
from sources.data.data_set import DataSet
from sources.data.features import Features
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.ffnn.gen_ffnn import GenerateFFNN

FRACTION_PREDICTION_LABELED = 0.8
NUM_EPOCHS_PER_CYCLE = 50

features = [Features.PreviousLocation, Features.AccessPointDetection, Features.Temperature, Features.Acceleration,
            Features.Heading, Features.Volume, Features.Light]
data = DataCompiler([DataSet.SimpleSquare], features, False)
# data = DataCompiler([DataSet.SimpleSquare, DataSet.LongRectangle, DataSet.RectangleWithRamp, DataSet.ManyCorners], features)

print("Saving data...")
with open("/home/shino/Uni/master_thesis/bin/evaluation_data.pkl", 'wb') as file:
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
for data_set_index in range(len(data.result_features_dt)):
    dt_next_cycle_features = dt_next_cycle_features + data.result_features_dt[data_set_index][0]
    dt_next_cycle_labels = dt_next_cycle_labels + data.result_labels_dt[data_set_index][0]
    knn_next_cycle_features = knn_next_cycle_features + data.result_features_knn[data_set_index][0]
    knn_next_cycle_labels = knn_next_cycle_labels + data.result_labels_knn[data_set_index][0]

model_knn = 0
model_dt = 0
for cycle in range(data.num_cycles - data.num_validation_cycles):
    print("Training cycle: {0}".format(cycle))
    print("Initializing...")
    # Reinitializing to make sure that there is no partial learning
    model_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, 16, 20)
    model_knn = GenerateFFNN(data.num_inputs, data.num_outputs)

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
    model_dt.fit(dt_data_features, dt_data_labels, 0.5)
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
        for i in range(1, int(len(dt_next_cycle_features) * FRACTION_PREDICTION_LABELED)):
            knn_pred = np.array(knn_prediction[i]).argmax() * (1 / data.num_outputs)
            if dt_prediction[i] != dt_next_cycle_features[i][0]:
                if dt_next_cycle_features[i][0] > 0:
                    dt_next_cycle_features[i][1] = dt_next_cycle_features[i][0]

            if abs(knn_pred - knn_next_cycle_features[i][0]) >= (1 / data.num_outputs):
                if knn_next_cycle_features[i][0] > 0:
                    knn_next_cycle_features[i][1] = knn_next_cycle_features[i][0]

            dt_next_cycle_features[i][0] = dt_prediction[i - 1]
            knn_next_cycle_features[i][0] = knn_pred

    print("")
    dt_acc = model_dt.evaluate_accuracy(model_dt.predict(dt_vs_features), dt_vs_labels)
    knn_acc = model_knn.evaluate_accuracy(model_knn.predict(knn_vs_features), knn_vs_labels)
    acc_per_cycle.append((dt_acc, knn_acc))

    print("Accuracy on validation set:")
    print("Accuracy DT: {0}".format(dt_acc))
    print("Accuracy KNN: {0}".format(knn_acc))

    print("")

print("Saving models...")
with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'wb') as file:
    pickle.dump(model_dt, file)

model_knn.save("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")

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
plt.savefig("/home/shino/Uni/master_thesis/bin/evaluation_acc_per_cycle.png")
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
plt.savefig("/home/shino/Uni/master_thesis/bin/evaluation_loss_acc_training.png")
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
plt.savefig("/home/shino/Uni/master_thesis/bin/evaluation_loss_acc_validation.png")
plt.clf()
plt.close(fig)
