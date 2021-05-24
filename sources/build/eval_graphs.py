import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from tensorflow import keras

from sources.config import BIN_FOLDER_PATH, NUM_CORES, parse_cmd_args, WITHOUT_PREVIOUS_EDGE
from sources.ffnn.gen_ffnn import GenerateFFNN
from sources.metric.compile_log import CompileLog
from sources.metric.graph_feature_importance import GraphFeatureImportance
from sources.metric.graph_location_distribution import GraphLocationDistribution
from sources.metric.graph_location_misclassified import GraphLocationMisclassified
from sources.metric.graph_location_misclassified_distribution import GraphLocationMisclassifiedDistribution
from sources.metric.graph_location_missclassification import GraphLocationMisclassification
from sources.metric.graph_path_segment_misclassified import GraphPathSegmentMisclassified
from sources.metric.graph_recognized_path_segment import GraphRecognizedPathSegment
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted
from sources.metric.graph_window_confidence import GraphWindowConfidence
from sources.metric.graph_window_confidence_not_zero import GraphWindowConfidenceNotZero
from sources.metric.graph_window_location_changes import GraphWindowLocationChanges
from sources.metric.log_metrics import LogMetrics

"""
This file utilizes the evaluated and generated models in order to generate metrics and graphs
that help to evaluate how good the models actually are
"""

encode_paths_between_as_location, dt_forest_size, dt_max_height, ffnn_num_hidden_layers, \
ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs, load_from_disk, \
pregen_path, evaluation_name, res_input_data_sets, pregen_anamoly_path = parse_cmd_args()


def generate_graphs(path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction, feature_name_map, evaluation_name,
                    is_anomaly, encode_paths_between_as_location):
    start_eval = time.time()

    # Loaded here cause it cant be pickled
    model_knn = 0
    extra_suffix = ""
    if is_anomaly:
        extra_suffix = "_anomaly"
        model_knn = keras.models.load_model(
            BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5", compile=False)
    else:
        model_knn = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5",
                                            compile=False)

    #GraphFeatureImportance(path, "evaluation" + extra_suffix, model_dt, model_knn, test_set_features_knn,
    #                       test_set_labels_knn,
    #                       test_set_features_dt, test_set_labels_dt, feature_name_map, is_anomaly)

    now = time.time()
    predicted_dt = model_dt.continued_predict(test_set_features_dt,
                                              encode_paths_between_as_location) if use_continued_prediction else model_dt.predict(
        test_set_features_dt)
    print("DT needed: {0} => ({1})".format(time.time() - now, use_continued_prediction))
    now = time.time()
    predicted_knn = GenerateFFNN.static_continued_predict(model_knn, test_set_features_knn,
                                                          num_outputs) if use_continued_prediction else model_knn.predict(
        test_set_features_knn)
    print("KNN needed: {0} => ({1})".format(time.time() - now, use_continued_prediction))

    GraphTrueVsPredicted(path, prefix + "_dt" + extra_suffix, True, test_set_labels_dt, num_outputs, predicted_dt)
    GraphTrueVsPredicted(path, prefix + "_knn" + extra_suffix, False, test_set_labels_knn, num_outputs, predicted_knn)

    if not is_anomaly:
        #GraphRecognizedPathSegment(path, prefix + "_dt", True, test_set_labels_dt, predicted_dt)
        #GraphRecognizedPathSegment(path, prefix + "_knn", False, test_set_labels_knn, predicted_knn)

        #GraphLocationMisclassified(path, prefix + "_dt", True, test_set_labels_dt, num_outputs, predicted_dt)
        #GraphLocationMisclassified(path, prefix + "_knn", False, test_set_labels_knn, num_outputs, predicted_knn)

        #GraphLocationMisclassification(path, prefix + "_dt", True, test_set_labels_dt, num_outputs, predicted_dt)
        #GraphLocationMisclassification(path, prefix + "_knn", False, test_set_labels_knn, num_outputs, predicted_knn)

        #GraphLocationMisclassifiedDistribution(path, prefix + "_dt", True, test_set_labels_dt, num_outputs,
        #                                       predicted_dt)
        #GraphLocationMisclassifiedDistribution(path, prefix + "_knn", False, test_set_labels_knn, num_outputs,
        #                                       predicted_knn)

        #GraphPathSegmentMisclassified(path, prefix + "_dt", True, test_set_labels_dt, predicted_dt)
        #GraphPathSegmentMisclassified(path, prefix + "_knn", False, test_set_labels_knn, predicted_knn)

        now = time.time()
        predicted_dt = model_dt.continued_predict_proba(
            test_set_features_dt,
            encode_paths_between_as_location) if use_continued_prediction else model_dt.predict_proba(
            test_set_features_dt)
        print("DT2 needed: {0} => ({1})".format(time.time() - now, use_continued_prediction))

        GraphWindowLocationChanges(path, prefix + "_dt", predicted_dt, True, encode_paths_between_as_location)
        GraphWindowLocationChanges(path, prefix + "_knn", predicted_knn, False, encode_paths_between_as_location)

        GraphWindowConfidence(path, prefix + "_dt", predicted_dt, True, encode_paths_between_as_location)
        GraphWindowConfidence(path, prefix + "_knn", predicted_knn, False, encode_paths_between_as_location)

        GraphWindowConfidenceNotZero(path, prefix + "_dt", predicted_dt, True, encode_paths_between_as_location)
        GraphWindowConfidenceNotZero(path, prefix + "_knn", predicted_knn, False, encode_paths_between_as_location)

        #LogMetrics(path, prefix + "_dt", predicted_dt)
        #LogMetrics(path, prefix + "_knn", predicted_knn)

        #CompileLog(path, prefix + "_dt")
        #CompileLog(path, prefix + "_knn")
    print("Whole evaluation: {0} => ({1})".format(time.time() - start_eval, use_continued_prediction))


with open(pregen_path, 'rb') as file:
    data = pickle.load(file)
    anomaly_file = open(pregen_anamoly_path, "rb")
    anomaly_data = pickle.load(anomaly_file)
    with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb') as file:
        model_dt = pickle.load(file)

        model_anomaly_dt = pickle.load(
            open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_anomaly_model.pkl", 'rb'))


        def glue_test_sets(features_dt, labels_dt, features_knn, labels_knn):
            new_set_dt = []
            new_labels_dt = []
            new_set_knn = []
            new_labels_knn = []
            for j in range(data.num_cycles):
                new_set_dt = new_set_dt + features_dt[j]
                new_labels_dt = new_labels_dt + labels_dt[j]
                new_set_knn = new_set_knn + features_knn[j]
                new_labels_knn = new_labels_knn + labels_knn[j]

            return new_set_dt, new_labels_dt, new_set_knn, new_labels_knn


        train_labels = []
        for i in range(len(data.result_labels_dt)):
            for cycle in range(data.num_cycles):
                train_labels = train_labels + data.result_labels_dt[i][cycle]

        GraphLocationDistribution(BIN_FOLDER_PATH + "/", train_labels)

        test_set_names = []
        test_sets_dt = []
        test_labels_dt = []
        test_sets_knn = []
        test_labels_knn = []

        # Parameter data sets
        """
        for i in range(len(data.test_labels_dt)):
            test_set_names.append(data.name_map_data_sets_test[i])
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(data.test_features_dt[i],
                                                                                    data.test_labels_dt[i],
                                                                                    data.test_features_knn[i],
                                                                                    data.test_labels_knn[i])

            test_sets_dt.append(np.asarray(new_set_dt).copy())
            test_labels_dt.append(np.asarray(new_labels_dt).copy())
            test_sets_knn.append(np.asarray(new_set_knn).copy())
            test_labels_knn.append(np.asarray(new_labels_knn).copy())

        # Faulty data sets
        for i in range(len(data.faulty_test_features_dt)):
            test_set_names.append(data.name_map_data_sets_faulty_test[i])
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(data.faulty_test_features_dt[i],
                                                                                    data.faulty_test_labels_dt[i],
                                                                                    data.faulty_test_features_knn[i],
                                                                                    data.faulty_test_labels_knn[i])

            test_sets_dt.append(np.asarray(new_set_dt).copy())
            test_labels_dt.append(np.asarray(new_labels_dt).copy())
            test_sets_knn.append(np.asarray(new_set_knn).copy())
            test_labels_knn.append(np.asarray(new_labels_knn).copy())
        """
        """
        # Anomaly data sets
        for i in range(1, 6):
            test_set_names.append(data.name_map_data_sets_temporary[i])
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(
                data.temporary_test_set_features_dt[i],
                data.temporary_test_set_labels_dt[i],
                data.temporary_test_set_features_knn[i],
                data.temporary_test_set_labels_knn[i])

            test_sets_dt.append(np.asarray(new_set_dt).copy())
            test_labels_dt.append(np.asarray(new_labels_dt).copy())
            test_sets_knn.append(np.asarray(new_set_knn).copy())
            test_labels_knn.append(np.asarray(new_labels_knn).copy())

        num_outputs = data.num_outputs
        name_map_features = data.name_map_features

        data = 0

        pool = Pool(NUM_CORES)
        workers = []
        for k in range(0, len(test_set_names)):
            print("Processing {0} of {1}...".format(k, len(test_set_names)))
            path = BIN_FOLDER_PATH + "/" + evaluation_name + "/" + test_set_names[k] + "/"
            # Create folder
            try:
                os.mkdir(path)
            except:
                pass

            try:
                os.mkdir(path + "evaluation/")
            except:
                pass

            # Valid set
            test_set_features_dt = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn = np.asarray(test_sets_knn[k]).copy()
            test_set_labels_dt = np.asarray(test_labels_dt[k]).copy()
            test_set_labels_knn = np.asarray(test_labels_knn[k]).copy()

            if not WITHOUT_PREVIOUS_EDGE:
                workers.append(pool.apply_async(generate_graphs,
                                                args=(path, "evaluation_continued", model_dt, test_set_features_dt,
                                                      test_set_features_knn,
                                                      test_set_labels_dt, test_set_labels_knn, num_outputs, True,
                                                      name_map_features,
                                                      evaluation_name, False, encode_paths_between_as_location,)))

            workers.append(pool.apply_async(generate_graphs,
                                            args=(path, "evaluation", model_dt, test_set_features_dt,
                                                  test_set_features_knn,
                                                  test_set_labels_dt, test_set_labels_knn, num_outputs, False,
                                                  name_map_features,
                                                  evaluation_name, False, encode_paths_between_as_location,)))

            while len(workers) >= NUM_CORES:
                for worker_index in range(len(workers)):
                    if workers[worker_index].ready():
                        workers.pop(worker_index)
                        break
                time.sleep(0.5)

        pool.close()
        pool.join()
        """

        data = 0

        path = BIN_FOLDER_PATH + "/" + evaluation_name + "/anomaly_model_test_set/"
        # Create folder
        try:
            os.mkdir(path)
        except:
            pass

        try:
            os.mkdir(path + "evaluation/")
        except:
            pass

        # Valid set
        test_set_features_dt = np.asarray(anomaly_data.test_features_dt).copy()
        test_set_features_knn = np.asarray(anomaly_data.test_features_knn).copy()
        test_set_labels_dt = np.asarray(anomaly_data.test_labels).copy()
        test_set_labels_knn = np.asarray(anomaly_data.test_labels).copy()

        generate_graphs(
        path, "evaluation", model_anomaly_dt, test_set_features_dt, test_set_features_knn,
        test_set_labels_dt, test_set_labels_knn, 2, False, ["Abweichung\nØStandortänderungen", "Abweichung\nØKlassifizierungs-\nwahrscheinlichkeit", "Topologieverletzung", "Standardabweichung\nTop 5 Klassifizierungen"],
        evaluation_name, True, encode_paths_between_as_location)