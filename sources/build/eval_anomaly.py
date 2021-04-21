import pickle
import sys
from multiprocessing import Pool

import numpy as np
from tensorflow import keras

from sources.config import BIN_FOLDER_PATH, NUM_CORES
from sources.decision_tree.ensemble_method import EnsembleMethod
from sources.decision_tree.gen_dt import GenerateDecisionTree
from sources.ffnn.gen_ffnn import GenerateFFNN

"""
In this file the anomaly model is trained. 
Requires the classification model to be trained before.
"""

# For some magical reason required by get_context method of multiprocessing
if __name__ == "__main__":
    _, evaluation_name = sys.argv
    NUM_EPOCHS_PER_CYCLE = 75
    WINDOW_SIZE = 25


    def calculate_anomaly_features_and_labels(predicted_dt, predicted_knn, data_set_index):
        res_features_dt = []
        res_features_knn = []
        res_labels_dt = []
        res_labels_knn = []

        location_changes_dt = []
        location_changes_knn = []
        confidence_dt = []
        confidence_knn = []
        amount_zero_loc_dt = []
        amount_zero_loc_knn = []
        current_location_dt = np.asarray(predicted_dt[0]).argmax()
        current_location_knn = np.asarray(predicted_knn[0]).argmax()
        for i in range(len(predicted_dt)):
            # Preparing the labels
            if data.temporary_test_set_raw_data[data_set_index].iloc[i]["is_anomaly"]:
                res_labels_dt.append(1)
                res_labels_knn.append([0, 1])
            else:
                res_labels_dt.append(0)
                res_labels_knn.append([1, 0])

            # Preparing the features
            features_dt = []
            features_knn = []

            # Location changes within window
            new_location_dt = np.asarray(predicted_dt[i]).argmax()
            if new_location_dt != current_location_dt:
                current_location_dt = new_location_dt
                location_changes_dt.append(1)
            else:
                location_changes_dt.append(0)

            if len(location_changes_dt) > WINDOW_SIZE:
                location_changes_dt.pop(0)

            new_location_knn = np.asarray(predicted_knn[i]).argmax()
            if new_location_knn != current_location_knn:
                current_location_knn = new_location_knn
                location_changes_knn.append(1)
            else:
                location_changes_knn.append(0)

            if len(location_changes_knn) > WINDOW_SIZE:
                location_changes_knn.pop(0)

            features_dt.append(sum(location_changes_dt))
            features_knn.append(sum(location_changes_knn) / WINDOW_SIZE)

            # Accumulated confidence
            confidence_dt.append(predicted_dt[i][new_location_dt])
            confidence_knn.append(predicted_knn[i][new_location_knn])

            if len(confidence_dt) > WINDOW_SIZE:
                confidence_dt.pop(0)

            if len(confidence_knn) > WINDOW_SIZE:
                confidence_knn.pop(0)

            features_dt.append(sum(confidence_dt))
            features_knn.append(sum(confidence_knn) / WINDOW_SIZE)

            # Current confidence
            features_dt.append(confidence_dt[-1])
            features_knn.append(confidence_knn[-1])

            # Change to previous confidence
            if len(confidence_dt) == 1:
                features_dt.append(0)
                features_knn.append(0)
            else:
                features_dt.append(abs(confidence_dt[-2] - confidence_dt[-1]))
                features_knn.append(abs(confidence_knn[-2] - confidence_knn[-1]))

            # Amount Zero Location
            amount_zero_loc_dt.append(1 if new_location_dt == 0 else 0)
            amount_zero_loc_knn.append(1 if new_location_knn == 0 else 0)

            if len(amount_zero_loc_dt) > WINDOW_SIZE:
                amount_zero_loc_dt.pop(0)

            if len(amount_zero_loc_knn) > WINDOW_SIZE:
                amount_zero_loc_knn.pop(0)

            features_dt.append(sum(amount_zero_loc_dt))
            features_knn.append(sum(amount_zero_loc_knn) / WINDOW_SIZE)

            res_features_dt.append(features_dt)
            res_features_knn.append(features_knn)

        return res_features_dt, res_labels_dt, res_features_knn, res_labels_knn


    def calculate_data_set(data_set_index, data, model_dt):
        model_knn = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5")
        af_dt = []
        af_knn = []
        al_dt = []
        al_knn = []

        af_dt_val = []
        af_knn_val = []
        al_dt_val = []
        al_knn_val = []

        # Training set
        data_set_features_dt = []
        data_set_features_knn = []
        for cycle in range(data.num_cycles - data.num_validation_cycles):
            data_set_features_dt = data_set_features_dt + data.temporary_test_set_features_dt[data_set_index][cycle]
            data_set_features_knn = data_set_features_knn + data.temporary_test_set_features_knn[data_set_index][cycle]

        predicted_dt = model_dt.continued_predict_proba(data_set_features_dt)
        predicted_knn = model_knn.continued_predict(data_set_features_knn)

        res_features_dt, res_labels_dt, res_features_knn, res_labels_knn = calculate_anomaly_features_and_labels(
            predicted_dt, predicted_knn, data_set_index)

        af_dt = af_dt + res_features_dt
        af_knn = af_knn + res_features_knn
        al_dt = al_dt + res_labels_dt
        al_knn = al_knn + res_labels_knn

        # Validation set
        data_set_features_dt = []
        data_set_features_knn = []
        for cycle in range(data.num_cycles - data.num_validation_cycles, data.num_cycles):
            data_set_features_dt = data_set_features_dt + data.temporary_test_set_features_dt[data_set_index][cycle]
            data_set_features_knn = data_set_features_knn + data.temporary_test_set_features_knn[data_set_index][cycle]

        predicted_dt = model_dt.continued_predict_proba(data_set_features_dt)
        predicted_knn = model_knn.continued_predict(data_set_features_knn)

        res_features_dt, res_labels_dt, res_features_knn, res_labels_knn = calculate_anomaly_features_and_labels(
            predicted_dt, predicted_knn, data_set_index)

        af_dt_val = af_dt_val + res_features_dt
        af_knn_val = af_knn_val + res_features_knn
        al_dt_val = al_dt_val + res_labels_dt
        al_knn_val = al_knn_val + res_labels_knn

        return af_dt, al_dt, af_dt_val, al_dt_val, af_knn, al_knn, af_knn_val, al_knn_val


    print("Loading data and models...")
    with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_data.pkl", 'rb') as file:
        data = pickle.load(file)
        with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb') as file:
            map_args = []
            model_dt = pickle.load(file)

            print("Generating Training and Validation Data...")
            args = []

            for data_set_index in range(len(data.temporary_test_set_features_dt)):
                args.append(data_set_index, data, model_dt)

            anomaly_features_dt = []
            anomaly_labels_dt = []
            anomaly_features_knn = []
            # TODO: binary_crossentropy + Sigmoid
            anomaly_labels_knn = []

            anomaly_features_dt_val = []
            anomaly_labels_dt_val = []
            anomaly_features_knn_val = []
            anomaly_labels_knn_val = []

            with Pool(processes=NUM_CORES) as pool:
                for res in pool.map(calculate_data_set, args):
                    af_dt, al_dt, af_dt_val, al_dt_val, af_knn, al_knn, af_knn_val, al_knn_val = res

                    anomaly_features_dt = anomaly_features_dt + af_dt
                    anomaly_labels_dt = anomaly_labels_dt + al_dt
                    anomaly_features_dt_val = anomaly_features_dt_val + af_dt_val
                    anomaly_labels_dt_val = anomaly_labels_dt_val + al_dt_val

                    anomaly_features_knn = anomaly_features_knn + af_knn
                    anomaly_labels_knn = anomaly_labels_knn + al_knn
                    anomaly_features_knn_val = anomaly_features_knn_val + af_knn_val
                    anomaly_labels_knn_val = anomaly_labels_knn_val + al_knn_val

            print("Training the anomaly detection models....")
            model_anomaly_dt = GenerateDecisionTree(EnsembleMethod.RandomForest, 8, 20)
            model_anomaly_knn = GenerateFFNN(5, 2, 1, 16, 75)

            model_anomaly_dt.fit(anomaly_features_dt, anomaly_labels_dt, 0.25)
            model_anomaly_knn.fit(anomaly_features_knn, anomaly_labels_knn, anomaly_features_knn_val,
                                  anomaly_labels_knn_val)

            print("Validating the models...")
            dt_acc = model_anomaly_dt.evaluate_accuracy(model_anomaly_dt.predict(anomaly_features_dt_val),
                                                        anomaly_labels_dt_val)
            knn_acc = model_anomaly_knn.evaluate_accuracy(model_anomaly_knn.predict(anomaly_features_knn_val),
                                                          anomaly_labels_knn_val)

            print("Accuracy DT: {0}".format(dt_acc))
            print("Accuracy KNN: {0}".format(knn_acc))

            print("Saving anomaly models...")
            with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_anomaly_model.pkl", 'wb') as file:
                pickle.dump(model_anomaly_dt, file)

            model_anomaly_knn.save(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5")

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
