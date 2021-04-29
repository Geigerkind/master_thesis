import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from sources.anomaly.topology_guesser import AnomalyTopologyGuesser
from sources.config import BIN_FOLDER_PATH, NUM_CORES, parse_cmd_args
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.ffnn.gen_ffnn import GenerateFFNN

"""
In this file the anomaly model is trained. 
Requires the classification model to be trained before.
"""

WITH_FEEDBACK_EDGE = False

# For some magical reason required by get_context method of multiprocessing
if __name__ == "__main__":
    encode_paths_between_as_location, dt_forest_size, dt_max_height, ffnn_num_hidden_layers, \
    ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs, load_from_disk, \
    pregen_path, evaluation_name, res_input_data_sets = parse_cmd_args()

    NUM_EPOCHS_PER_CYCLE = 150
    WINDOW_SIZE = 35


    def calculate_anomaly_features_and_labels(predicted_dt, predicted_knn, data_set_index, real_labels, data):
        global WITH_FEEDBACK_EDGE

        res_features_dt = []
        res_features_knn = []
        res_labels = []

        location_changes_dt = []
        location_changes_no_anomaly_dt = []
        location_changes_knn = []
        location_changes_no_anomaly_knn = []
        confidence_dt = []
        confidence_no_anomaly_dt = []
        confidence_knn = []
        confidence_no_anomaly_knn = []
        amount_zero_loc_dt = []
        amount_zero_loc_knn = []
        current_location_dt = np.asarray(predicted_dt[0]).argmax()
        current_location_knn = np.asarray(predicted_knn[0]).argmax()

        distinct_locations_dt = [0, 0]
        distinct_locations_knn = [0, 0]

        num_outputs = len(predicted_dt[0])
        topology_guesser_dt = AnomalyTopologyGuesser(data.location_neighbor_graph)
        topology_guesser_knn = AnomalyTopologyGuesser(data.location_neighbor_graph)

        real_previous_location = 0
        # Find first non zero location and take the previous of it
        for label in real_labels:
            if label > 0:
                # Otherwise its anomaly data
                if label in data.location_neighbor_graph:
                    real_previous_location = data.location_neighbor_graph[label][0]
                break

        for i in range(len(predicted_dt)):
            # Preparing the labels
            if data.temporary_test_set_raw_data[data_set_index].iloc[i]["is_anomaly"]:
                res_labels.append(1)
            else:
                res_labels.append(0)

            if real_labels[i] > 0 and real_labels[i] != real_previous_location:
                real_previous_location = real_labels[i]

            new_location_dt = np.asarray(predicted_dt[i]).argmax()
            new_location_knn = np.asarray(predicted_knn[i]).argmax()

            # Preparing the features
            features_dt = []
            features_knn = []

            # Previous value.
            # Just like the other model this will be gradually overwritten
            if WITH_FEEDBACK_EDGE:
                if i > 0:
                    features_dt.append(int(data.temporary_test_set_raw_data[data_set_index].iloc[i - 1]["is_anomaly"]))
                    features_knn.append(int(data.temporary_test_set_raw_data[data_set_index].iloc[i - 1]["is_anomaly"]))
                else:
                    features_dt.append(0)
                    features_knn.append(0)

            if 0 < new_location_dt != distinct_locations_dt[-1]:
                distinct_locations_dt.append(new_location_dt)
                distinct_locations_knn.append(new_location_knn)

            # Location changes within window
            if new_location_dt != current_location_dt:
                current_location_dt = new_location_dt
                location_changes_dt.append(1)
            else:
                location_changes_dt.append(0)

            if new_location_knn != current_location_knn:
                current_location_knn = new_location_knn
                location_changes_knn.append(1)
            else:
                location_changes_knn.append(0)

            # features_dt.append(sum(location_changes_dt[-WINDOW_SIZE:]))
            # features_knn.append(sum(location_changes_knn[-WINDOW_SIZE:]) / WINDOW_SIZE)

            # Accumulated confidence
            confidence_dt.append(predicted_dt[i][new_location_dt])
            confidence_knn.append(predicted_knn[i][new_location_knn])

            # features_dt.append(sum(confidence_dt[-WINDOW_SIZE:]))
            # features_knn.append(sum(confidence_knn[-WINDOW_SIZE:]) / WINDOW_SIZE)

            # Current confidence
            # features_dt.append(confidence_dt[-1])
            # features_knn.append(confidence_knn[-1])

            # Change to previous confidence
            # if len(confidence_dt) == 1:
            #    features_dt.append(0)
            #    features_knn.append(0)
            # else:
            #    features_dt.append(abs(confidence_dt[-2] - confidence_dt[-1]))
            #    features_knn.append(abs(confidence_knn[-2] - confidence_knn[-1]))

            # Amount Zero Location
            # amount_zero_loc_dt.append(1 if new_location_dt == 0 else 0)
            # amount_zero_loc_knn.append(1 if new_location_knn == 0 else 0)
            # features_dt.append(sum(amount_zero_loc_dt[-WINDOW_SIZE:]))
            # features_knn.append(sum(amount_zero_loc_knn[-WINDOW_SIZE:]) / WINDOW_SIZE)

            # Deviation no anomaly
            if i > 0 and not data.temporary_test_set_raw_data[data_set_index].iloc[i - 1]["is_anomaly"]:
                location_changes_no_anomaly_dt.append(location_changes_dt[-2])
                location_changes_no_anomaly_knn.append(location_changes_knn[-2])
                confidence_no_anomaly_dt.append(confidence_dt[-2])
                confidence_no_anomaly_knn.append(confidence_knn[-2])

            # window location changes deviation to the average
            # features_dt.append(abs((sum(location_changes_dt) / len(location_changes_dt)) - (sum(location_changes_dt[-WINDOW_SIZE:]) / len(location_changes_dt[-WINDOW_SIZE:]))))
            # features_knn.append(abs((sum(location_changes_knn) / len(location_changes_knn)) - (sum(location_changes_knn[-WINDOW_SIZE:]) / len(location_changes_knn[-WINDOW_SIZE:]))))
            features_dt.append(abs(
                (sum(location_changes_no_anomaly_dt) / max(len(location_changes_no_anomaly_dt), 1)) - (
                        sum(location_changes_dt[-WINDOW_SIZE:]) / max(len(location_changes_dt[-WINDOW_SIZE:]), 1))))
            features_knn.append(abs(
                (sum(location_changes_no_anomaly_knn) / max(len(location_changes_no_anomaly_knn), 1)) - (
                        sum(location_changes_knn[-WINDOW_SIZE:]) / max(len(location_changes_knn[-WINDOW_SIZE:]),
                                                                       1))))

            # window confidence changes deviation to the average
            # features_dt.append(abs((sum(confidence_dt) / len(confidence_dt)) - (sum(confidence_dt[-WINDOW_SIZE:]) / len(confidence_dt[-WINDOW_SIZE:]))))
            # features_knn.append(abs((sum(confidence_knn) / len(confidence_knn)) - (sum(confidence_knn[-WINDOW_SIZE:]) / len(confidence_knn[-WINDOW_SIZE:]))))
            features_dt.append(abs((sum(confidence_no_anomaly_dt) / max(len(confidence_no_anomaly_dt), 1)) - (
                    sum(confidence_dt[-WINDOW_SIZE:]) / max(len(confidence_dt[-WINDOW_SIZE:]), 1))))
            features_knn.append(abs((sum(confidence_no_anomaly_knn) / max(len(confidence_no_anomaly_knn), 1)) - (
                    sum(confidence_knn[-WINDOW_SIZE:]) / max(len(confidence_knn[-WINDOW_SIZE:]), 1))))

            # Last distinct location and current location to learn the mapping
            # I choose to use random values in case its an anomaly,
            # because I dont want to learn it a particular mapping there
            """
            if data.temporary_test_set_raw_data[data_set_index].iloc[i]["is_anomaly"] or (
            not (real_previous_location in data.location_neighbor_graph)):
                features_dt.append(random.randint(0, num_outputs - 1))
                features_dt.append(random.randint(0, num_outputs - 1))
                features_knn.append(random.randint(0, num_outputs - 1) / (num_outputs - 1))
                features_knn.append(random.randint(0, num_outputs - 1) / (num_outputs - 1))
            else:
                features_dt.append(data.location_neighbor_graph[real_previous_location][0])
                features_knn.append(data.location_neighbor_graph[real_previous_location][0] / (num_outputs - 1))
                features_dt.append(real_labels[i])
                features_knn.append(real_labels[i] / (num_outputs - 1))
            """

            # Topology Guesser As input, its worth a try!
            if new_location_dt == 0:
                features_dt.append(int(topology_guesser_dt.predict(distinct_locations_dt[-1], 0)))
            else:
                features_dt.append(int(topology_guesser_dt.predict(distinct_locations_dt[-2], new_location_dt)))

            if new_location_knn == 0:
                features_knn.append(int(topology_guesser_knn.predict(distinct_locations_knn[-1], 0)))
            else:
                features_knn.append(int(topology_guesser_knn.predict(distinct_locations_knn[-2], new_location_knn)))

            # Express confidence by the standard deviation of the top 5 guesses
            features_dt.append(np.asarray(sorted(predicted_dt[i], key=lambda i: i, reverse=True)[:5]).std())
            features_knn.append(np.asarray(sorted(predicted_knn[i], key=lambda i: i, reverse=True)[:5]).std())

            res_features_dt.append(features_dt)
            res_features_knn.append(features_knn)

        return res_features_dt, res_labels, res_features_knn


    def calculate_data_set(args):
        data_set_index, data, model_dt = args

        model_knn = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5")
        af_dt = []
        af_knn = []
        al = []

        af_dt_val = []
        af_knn_val = []
        al_val = []

        # Training set
        data_set_features_dt = []
        data_set_features_knn = []
        data_set_labels = []
        for cycle in range(data.num_cycles - data.num_validation_cycles + 1):
            data_set_features_dt = data_set_features_dt + data.temporary_test_set_features_dt[data_set_index][cycle]
            data_set_features_knn = data_set_features_knn + data.temporary_test_set_features_knn[data_set_index][cycle]
            data_set_labels = data_set_labels + data.temporary_test_set_labels_dt[data_set_index][cycle]

        predicted_dt = model_dt.continued_predict_proba(data_set_features_dt)
        predicted_knn = GenerateFFNN.static_continued_predict(model_knn, data_set_features_knn, data.num_outputs)

        res_features_dt, res_labels, res_features_knn = calculate_anomaly_features_and_labels(
            predicted_dt, predicted_knn, data_set_index, data_set_labels, data)

        af_dt = af_dt + res_features_dt
        af_knn = af_knn + res_features_knn
        al = al + res_labels

        # Validation set
        data_set_features_dt = []
        data_set_features_knn = []
        data_set_labels = []
        for cycle in range(data.num_cycles - data.num_validation_cycles + 1, data.num_cycles):
            data_set_features_dt = data_set_features_dt + data.temporary_test_set_features_dt[data_set_index][cycle]
            data_set_features_knn = data_set_features_knn + data.temporary_test_set_features_knn[data_set_index][cycle]
            data_set_labels = data_set_labels + data.temporary_test_set_labels_dt[data_set_index][cycle]

        predicted_dt_val = model_dt.continued_predict_proba(data_set_features_dt)
        predicted_knn_val = GenerateFFNN.static_continued_predict(model_knn, data_set_features_knn, data.num_outputs)

        res_features_dt, res_labels, res_features_knn = calculate_anomaly_features_and_labels(
            predicted_dt_val, predicted_knn_val, data_set_index, data_set_labels, data)

        af_dt_val = af_dt_val + res_features_dt
        af_knn_val = af_knn_val + res_features_knn
        al_val = al_val + res_labels

        return af_dt, al, af_dt_val, al_val, af_knn, af_knn_val, predicted_dt_val, predicted_knn_val


    class DumpAnomalyData:
        def __init__(self, t_f_dt, t_f_knn, t_f_dt_val, t_f_knn_val, t_l, t_l_val, test_f_dt, test_f_knn, test_l):
            self.train_features_dt = t_f_dt
            self.train_features_knn = t_f_knn
            self.train_features_dt_val = t_f_dt_val
            self.train_features_knn_val = t_f_knn_val
            self.train_labels = t_l
            self.train_labels_val = t_l_val

            self.test_features_dt = test_f_dt
            self.test_features_knn = test_f_knn
            self.test_labels = test_l


    print("Loading data and models...")
    with open(pregen_path, 'rb') as file:
        data = pickle.load(file)
        with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb') as file:
            map_args = []
            model_dt = pickle.load(file)

            print("Generating Training and Validation Data...")
            args = []

            for data_set_index in range(1, len(data.temporary_test_set_labels_dt)):
                args.append([data_set_index, data, model_dt])

            anomaly_features_dt = []
            anomaly_features_knn = []
            anomaly_labels = []

            anomaly_features_dt_val = []
            anomaly_features_knn_val = []
            anomaly_labels_val = []

            predicted_dt_val = []
            predicted_knn_val = []

            anomaly_features_dt_test = []
            anomaly_features_knn_test = []
            anomaly_labels_test = []

            with Pool(processes=NUM_CORES) as pool:
                result = pool.map(calculate_data_set, args)
                for i in [1, 2]:
                    af_dt, al, af_dt_val, al_val, af_knn, af_knn_val, pred_dt_val, pred_knn_val = result[i]

                    anomaly_features_dt = anomaly_features_dt + af_dt
                    anomaly_labels = anomaly_labels + al
                    anomaly_features_dt_val = anomaly_features_dt_val + af_dt_val
                    anomaly_labels_val = anomaly_labels_val + al_val

                    anomaly_features_knn = anomaly_features_knn + af_knn
                    anomaly_features_knn_val = anomaly_features_knn_val + af_knn_val

                    predicted_dt_val = predicted_dt_val + pred_dt_val
                    predicted_knn_val = predicted_knn_val + pred_knn_val
                for i in range(3, len(data.temporary_test_set_labels_dt)):
                    af_dt, al, af_dt_val, al_val, af_knn, af_knn_val, pred_dt_val, pred_knn_val = result[i]

                    anomaly_features_dt_test = anomaly_features_dt_test + af_dt + af_dt_val
                    anomaly_features_knn_test = anomaly_features_knn_test + af_knn + af_knn_val
                    anomaly_labels_test = anomaly_labels_test + al + al_val

            print("Training the anomaly detection models....")
            model_anomaly_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, 8, 20)
            model_anomaly_knn = GenerateFFNN(len(anomaly_features_knn[0]), 2, 1, 32, NUM_EPOCHS_PER_CYCLE, True)

            if not WITH_FEEDBACK_EDGE:
                model_anomaly_dt.fit(anomaly_features_dt, anomaly_labels, 0.25)
                model_anomaly_knn.fit(anomaly_features_knn, anomaly_labels, anomaly_features_knn_val,
                                      anomaly_labels_val)
            else:
                NUM_TRAINING_CYCLES = 10
                NUM_WARMUP_CYCLES = 2
                INITIAL_PREDICT_FRACTION = 0.5
                CYCLE_WHERE_EVERYTHING_IS_PREDICTED = 10

                af_dt = np.array_split(anomaly_features_dt, NUM_TRAINING_CYCLES + NUM_WARMUP_CYCLES)
                af_knn = np.array_split(anomaly_features_knn, NUM_TRAINING_CYCLES + NUM_WARMUP_CYCLES)
                al = np.array_split(anomaly_labels, NUM_TRAINING_CYCLES + NUM_WARMUP_CYCLES)

                train_af_dt = []
                train_af_knn = []
                train_al = []
                for cycle in range(NUM_WARMUP_CYCLES):
                    train_af_dt = train_af_dt + af_dt[cycle].tolist()
                    train_af_knn = train_af_knn + af_knn[cycle].tolist()
                    train_al = train_al + al[cycle].tolist()

                model_anomaly_dt.fit(train_af_dt, train_al, 0.25)
                model_anomaly_knn.fit(train_af_knn, train_al, anomaly_features_knn_val, anomaly_labels_val)

                for cycle in range(NUM_WARMUP_CYCLES, NUM_TRAINING_CYCLES + NUM_WARMUP_CYCLES):
                    print("Training cycle {0} of {1}...".format(cycle + 1, NUM_TRAINING_CYCLES + NUM_WARMUP_CYCLES))
                    # Predict a fraction of the dataset
                    fraction_to_predict = min(1.0, ((1.0 - INITIAL_PREDICT_FRACTION) / (
                            (CYCLE_WHERE_EVERYTHING_IS_PREDICTED - NUM_WARMUP_CYCLES) ** 2)) * (
                                                      (cycle - NUM_WARMUP_CYCLES) ** 2) + INITIAL_PREDICT_FRACTION)

                    # Of that fraction we pick samples randomly
                    permutation = np.random.permutation(len(af_dt[cycle]))
                    predictions_dt = model_anomaly_dt.predict(af_dt[cycle])
                    predictions_knn = model_anomaly_knn.predict(af_knn[cycle])
                    for perm_index in range(0, int(len(af_dt[cycle]) * fraction_to_predict)):
                        i = permutation[perm_index]
                        if i == 0:
                            continue

                        af_dt[cycle][i][0] = predictions_dt[i - 1]
                        af_knn[cycle][i][0] = int(predictions_knn[i - 1][0] >= 0.5)

                    train_af_dt = train_af_dt + af_dt[cycle].tolist()
                    train_af_knn = train_af_knn + af_knn[cycle].tolist()
                    train_al = train_al + al[cycle].tolist()

                    model_anomaly_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, 8, 20)
                    model_anomaly_knn = GenerateFFNN(len(anomaly_features_knn[0]), 2, 1, 32, NUM_EPOCHS_PER_CYCLE, True)

                    model_anomaly_dt.fit(train_af_dt, train_al, 0.25)
                    model_anomaly_knn.fit(train_af_knn, train_al, anomaly_features_knn_val, anomaly_labels_val)

            print("Validating the models...")
            # TODO: Adjust features from validation set to use predicted values for the window deviation
            dt_acc = model_anomaly_dt.evaluate_accuracy(model_anomaly_dt.predict(anomaly_features_dt_val),
                                                        anomaly_labels_val)
            knn_acc = model_anomaly_knn.evaluate_accuracy(model_anomaly_knn.predict(anomaly_features_knn_val),
                                                          anomaly_labels_val)

            print("Accuracy DT: {0}".format(dt_acc))
            print("Accuracy KNN: {0}".format(knn_acc))

            acc_always_true = model_anomaly_dt.evaluate_accuracy([1 for _ in range(len(anomaly_labels_val))],
                                                                 anomaly_labels_val)
            acc_always_false = model_anomaly_dt.evaluate_accuracy([0 for _ in range(len(anomaly_labels_val))],
                                                                  anomaly_labels_val)
            print("Accuracy always True: {0}".format(acc_always_true))
            print("Accuracy always False: {0}".format(acc_always_false))

            # Prepare previous distinct locations for guessed locations
            previous_distinct_locations_dt = [0, 0]
            previous_distinct_locations_knn = [0, 0]

            res_previous_distinct_locations_dt = []
            res_previous_distinct_locations_knn = []
            for i in range(len(predicted_dt_val)):
                prediction_dt = np.asarray(predicted_dt_val[i]).argmax()
                prediction_knn = np.asarray(predicted_knn_val[i]).argmax()

                if 0 < prediction_dt != previous_distinct_locations_dt[-1]:
                    previous_distinct_locations_dt.append(prediction_dt)
                    previous_distinct_locations_dt.pop(0)

                if 0 < prediction_knn != previous_distinct_locations_knn[-1]:
                    previous_distinct_locations_knn.append(prediction_knn)
                    previous_distinct_locations_knn.pop(0)

                if prediction_dt == 0:
                    res_previous_distinct_locations_dt.append(previous_distinct_locations_dt[-1])
                else:
                    res_previous_distinct_locations_dt.append(previous_distinct_locations_dt[-2])

                if prediction_knn == 0:
                    res_previous_distinct_locations_knn.append(previous_distinct_locations_knn[-1])
                else:
                    res_previous_distinct_locations_knn.append(previous_distinct_locations_knn[-2])

            topology_guesser_dt = AnomalyTopologyGuesser(data.location_neighbor_graph)
            topology_guesser_knn = AnomalyTopologyGuesser(data.location_neighbor_graph)
            guessed_dt = []
            guessed_knn = []
            for i in range(len(res_previous_distinct_locations_dt)):
                guessed_dt.append(int(topology_guesser_dt.predict(res_previous_distinct_locations_dt[i],
                                                                  np.asarray(predicted_dt_val[i]).argmax())))
                guessed_knn.append(int(topology_guesser_knn.predict(res_previous_distinct_locations_knn[i],
                                                                    np.asarray(predicted_knn_val[i]).argmax())))

            acc_top_dt = model_anomaly_dt.evaluate_accuracy(guessed_dt, anomaly_labels_val)
            acc_top_knn = model_anomaly_dt.evaluate_accuracy(guessed_knn, anomaly_labels_val)
            print("Accuracy Anomaly Topology Guesser (DT): {0}".format(acc_top_dt))
            print("Accuracy Anomaly Topology Guesser (KNN): {0}".format(acc_top_knn))

            print("Saving anomaly models...")
            with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_anomaly_model.pkl", 'wb') as file:
                pickle.dump(model_anomaly_dt, file)

            model_anomaly_knn.save(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5")

            print("")
            print("Saving processed features and labels...")
            with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_anomaly_data.pkl", 'wb') as file:
                anomaly_data = DumpAnomalyData(anomaly_features_dt, anomaly_features_knn, anomaly_features_dt_val,
                                               anomaly_features_knn_val, anomaly_labels, anomaly_labels_val,
                                               anomaly_features_dt_test, anomaly_features_knn_test, anomaly_labels_test)
                pickle.dump(anomaly_data, file)

            print("")

            print("Generating fancy plots...")
            knn_hist = model_anomaly_knn.get_history()
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
            plt.savefig(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_anomaly_loss_acc_training.png")
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
            plt.savefig(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_anomaly_loss_acc_validation.png")
            plt.clf()
            plt.close(fig)
