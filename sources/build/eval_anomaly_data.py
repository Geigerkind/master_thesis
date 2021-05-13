import pickle

import numpy as np
from tensorflow import keras

from sources.ffnn.gen_ffnn import GenerateFFNN
from sources.anomaly.topology_guesser import AnomalyTopologyGuesser
from sources.config import BIN_FOLDER_PATH


def evaluate_accuracy(prediction, reality):
    correct = 0
    correct_true = 0
    correct_false = 0
    total_true = 0
    total_false = 0
    for i in range(len(prediction)):
        if reality[i] == 1:
            total_true = total_true + 1
        else:
            total_false = total_false + 1

        if prediction[i] == reality[i]:
            correct = correct + 1
            if reality[i] == 1:
                correct_true = correct_true + 1
            else:
                correct_false = correct_false + 1

    return correct / len(prediction), correct_true / total_true, correct_false / total_false


if __name__ == "__main__":
    dt_model_configs = [
        (0, 16, 8, 1, 16, 75, "1"),
        (1, 8, 32, 1, 32, 75, "1"),
        (0, 16, 8, 1, 16, 75, "12"),
        (0, 8, 32, 1, 32, 75, "123"),
        (1, 16, 16, 1, 32, 75, "12"),
        (1, 8, 32, 1, 32, 75, "123"),
        (0, 8, 32, 1, 32, 75, "1234"),
        (1, 16, 16, 1, 32, 75, "1234"),
    ]

    knn_model_configs = [
        (0, 16, 8, 1, 16, 75, "1"),
        (1, 16, 16, 1, 32, 75, "1"),
        (0, 16, 64, 1, 128, 75, "12"),
        (0, 16, 32, 1, 64, 75, "123"),
        (1, 16, 32, 2, 32, 75, "12"),
        (1, 16, 16, 1, 32, 75, "123"),
        (0, 16, 64, 1, 128, 75, "1234"),
        (1, 64, 32, 8, 32, 75, "1234"),
    ]

    print("Evaluating DTs...")
    count = 0
    result = []
    for dt_config in dt_model_configs:
        evaluation_name = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(dt_config[0], dt_config[1], dt_config[2],
                                                                              dt_config[3], dt_config[4], dt_config[5],
                                                                              dt_config[6])
        data = pickle.load(
            open(BIN_FOLDER_PATH + "/pregen_data/data_{0}_{1}.pkl".format(dt_config[0], dt_config[6]), "rb"))
        anomaly_data = pickle.load(open(BIN_FOLDER_PATH + "/pregen_data/data_anamoly_{0}_{1}.pkl".format(dt_config[0], dt_config[6]), "rb"))
        model_dt = pickle.load(open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb'))
        model_anomaly_dt = pickle.load(
            open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_anomaly_model.pkl", 'rb'))

        #data_set_features = []
        #for cycle in range(0, data.num_cycles):
        #    data_set_features = data_set_features + data.temporary_test_set_features_dt[2][cycle]

        #model_location_predictions = model_dt.continued_predict(data_set_features)
        model_location_predictions = [np.asarray(x).argmax() for x in anomaly_data.predicted_dt_test]

        model_predictions = model_anomaly_dt.predict(anomaly_data.test_features_dt)
        acc, acc_given_true, acc_given_false = evaluate_accuracy(model_predictions, anomaly_data.test_labels)
        acc_true, _, _ = evaluate_accuracy([1 for _ in range(len(anomaly_data.test_labels))],
                                                      anomaly_data.test_labels)
        acc_false, _ , _ = evaluate_accuracy([0 for _ in range(len(anomaly_data.test_labels))],
                                                       anomaly_data.test_labels)

        topology_guesser = AnomalyTopologyGuesser(data.location_neighbor_graph)
        guessed = []
        previous_distinct_locations = [0, 0]
        res_previous_distinct_locations = []
        for prediction in model_location_predictions:
            if 0 < prediction != previous_distinct_locations[-1]:
                previous_distinct_locations.append(prediction)
                previous_distinct_locations.pop(0)

            if prediction == 0:
                res_previous_distinct_locations.append(previous_distinct_locations[-1])
            else:
                res_previous_distinct_locations.append(previous_distinct_locations[-2])

        for i in range(len(res_previous_distinct_locations)):
            prediction_dt = model_location_predictions[i]
            guessed.append(
                int(topology_guesser.predict(res_previous_distinct_locations[i], prediction_dt)))

        acc_top, _, _ = evaluate_accuracy(guessed, anomaly_data.test_labels)

        print(count, " => ", (acc, acc_true, acc_false, acc_top, acc_given_true, acc_given_false))
        result.append([(acc, acc_true, acc_false, acc_top, acc_given_true, acc_given_false)])
        count = count + 1

    print("Evaluating FFNNs...")
    count = 0
    for knn_config in knn_model_configs:
        evaluation_name = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(knn_config[0], knn_config[1],
                                                                              knn_config[2],
                                                                              knn_config[3], knn_config[4],
                                                                              knn_config[5],
                                                                              knn_config[6])
        data = pickle.load(
            open(BIN_FOLDER_PATH + "/pregen_data/data_{0}_{1}.pkl".format(knn_config[0], knn_config[6]), "rb"))
        anomaly_data = pickle.load(open(BIN_FOLDER_PATH + "/pregen_data/data_anamoly_{0}_{1}.pkl".format(knn_config[0], knn_config[6]), "rb"))
        model_knn = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5",
                                            compile=False)
        model_anomaly_knn = keras.models.load_model(
            BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5", compile=False)

        #data_set_features = []
        #for cycle in range(0, data.num_cycles):
        #    data_set_features = data_set_features + data.temporary_test_set_features_knn[2][cycle]

        #model_location_predictions = GenerateFFNN.continued_predict(model_knn, np.asarray(data_set_features))
        model_location_predictions = anomaly_data.predicted_knn_test

        model_predictions = model_anomaly_knn.predict(np.asarray(anomaly_data.test_features_knn))
        model_predictions_distinct = [np.asarray(x).argmax() for x in model_predictions]

        acc, acc_given_true, acc_given_false = evaluate_accuracy(model_predictions_distinct, anomaly_data.test_labels)
        acc_true, _, _ = evaluate_accuracy([1 for _ in range(len(anomaly_data.test_labels))],
                                           anomaly_data.test_labels)
        acc_false, _, _ = evaluate_accuracy([0 for _ in range(len(anomaly_data.test_labels))],
                                            anomaly_data.test_labels)

        topology_guesser = AnomalyTopologyGuesser(data.location_neighbor_graph)
        guessed = []
        previous_distinct_locations = [0, 0]
        res_previous_distinct_locations = []
        for prediction_arr in model_location_predictions:
            prediction = np.asarray(prediction_arr).argmax()
            if 0 < prediction != previous_distinct_locations[-1]:
                previous_distinct_locations.append(prediction)
                previous_distinct_locations.pop(0)

            if prediction == 0:
                res_previous_distinct_locations.append(previous_distinct_locations[-1])
            else:
                res_previous_distinct_locations.append(previous_distinct_locations[-2])

        for i in range(len(res_previous_distinct_locations)):
            prediction = np.asarray(model_location_predictions[i]).argmax()
            guessed.append(
                int(topology_guesser.predict(res_previous_distinct_locations[i], prediction)))

        acc_top, _, _ = evaluate_accuracy(guessed, anomaly_data.test_labels)

        print(count, " => ", (acc, acc_true, acc_false, acc_top, acc_given_true, acc_given_false))
        result[count].append((acc, acc_true, acc_false, acc_top, acc_given_true, acc_given_false))
        count = count + 1

    for i in range(6):
        print("{0:.2f} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f}".format(100 * result[0][0][i], 100 * result[1][0][i], 100 * result[2][0][i], 100 * result[3][0][i], 100 * result[4][0][i], 100 * result[5][0][i], 100 * result[6][0][i], 100 * result[7][0][i]))
    print("{0:.2f} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f}".format(100 * result[0][1][0], 100 * result[1][1][0], 100 * result[2][1][0], 100 * result[3][1][0], 100 * result[4][1][0], 100 * result[5][1][0], 100 * result[6][1][0], 100 * result[7][1][0]))
    print("{0:.2f} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f}".format(100 * result[0][1][3], 100 * result[1][1][3], 100 * result[2][1][3], 100 * result[3][1][3], 100 * result[4][1][3], 100 * result[5][1][3], 100 * result[6][1][3], 100 * result[7][1][3]))
    print("{0:.2f} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f}".format(100 * result[0][1][4], 100 * result[1][1][4], 100 * result[2][1][4], 100 * result[3][1][4], 100 * result[4][1][4], 100 * result[5][1][4], 100 * result[6][1][4], 100 * result[7][1][4]))
    print("{0:.2f} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f}".format(100 * result[0][1][5], 100 * result[1][1][5], 100 * result[2][1][5], 100 * result[3][1][5], 100 * result[4][1][5], 100 * result[5][1][5], 100 * result[6][1][5], 100 * result[7][1][5]))