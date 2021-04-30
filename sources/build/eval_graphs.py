import os
import pickle
import random
from multiprocessing import Pool

import numpy as np
from tensorflow import keras

from sources.config import BIN_FOLDER_PATH, NUM_CORES, parse_cmd_args
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
pregen_path, evaluation_name, res_input_data_sets = parse_cmd_args()


def generate_graphs(path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction, feature_name_map, evaluation_name,
                    is_anomaly):
    # Loaded here cause it cant be pickled
    model_knn = 0
    extra_suffix = ""
    if is_anomaly:
        extra_suffix = "_anomaly"
        model_knn = keras.models.load_model(
            BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5")
    else:
        model_knn = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5")

    GraphFeatureImportance(path, "evaluation", model_dt, model_knn, test_set_features_knn, test_set_labels_knn,
                           test_set_features_dt, test_set_labels_dt, feature_name_map)

    predicted_dt = model_dt.continued_predict(test_set_features_dt) if use_continued_prediction else model_dt.predict(
        test_set_features_dt)
    predicted_knn = GenerateFFNN.static_continued_predict(model_knn, test_set_features_knn,
                                                          num_outputs) if use_continued_prediction else model_knn.predict(
        test_set_features_knn)

    GraphTrueVsPredicted(path, prefix + "_dt" + extra_suffix, True, test_set_labels_dt, num_outputs, predicted_dt)
    GraphTrueVsPredicted(path, prefix + "_knn" + extra_suffix, False, test_set_labels_knn, num_outputs, predicted_knn)

    if not is_anomaly:
        GraphRecognizedPathSegment(path, prefix + "_dt", True, test_set_labels_dt, predicted_dt)
        GraphRecognizedPathSegment(path, prefix + "_knn", False, test_set_labels_knn, predicted_knn)

        GraphLocationMisclassified(path, prefix + "_dt", True, test_set_labels_dt, num_outputs, predicted_dt)
        GraphLocationMisclassified(path, prefix + "_knn", False, test_set_labels_knn, num_outputs, predicted_knn)

        GraphLocationMisclassification(path, prefix + "_dt", True, test_set_labels_dt, num_outputs, predicted_dt)
        GraphLocationMisclassification(path, prefix + "_knn", False, test_set_labels_knn, num_outputs, predicted_knn)

        GraphLocationMisclassifiedDistribution(path, prefix + "_dt", True, test_set_labels_dt, num_outputs,
                                               predicted_dt)
        GraphLocationMisclassifiedDistribution(path, prefix + "_knn", False, test_set_labels_knn, num_outputs,
                                               predicted_knn)

        GraphPathSegmentMisclassified(path, prefix + "_dt", True, test_set_labels_dt, predicted_dt)
        GraphPathSegmentMisclassified(path, prefix + "_knn", False, test_set_labels_knn, predicted_knn)

        predicted_dt = model_dt.continued_predict_proba(
            test_set_features_dt) if use_continued_prediction else model_dt.predict_proba(
            test_set_features_dt)

        GraphWindowLocationChanges(path, prefix + "_dt", predicted_dt)
        GraphWindowLocationChanges(path, prefix + "_knn", predicted_knn)

        GraphWindowConfidence(path, prefix + "_dt", predicted_dt)
        GraphWindowConfidence(path, prefix + "_knn", predicted_knn)

        GraphWindowConfidenceNotZero(path, prefix + "_dt", predicted_dt)
        GraphWindowConfidenceNotZero(path, prefix + "_knn", predicted_knn)

        LogMetrics(path, prefix + "_dt", predicted_dt)
        LogMetrics(path, prefix + "_knn", predicted_knn)

        CompileLog(path, prefix + "_dt")
        CompileLog(path, prefix + "_knn")


def exec_gen_graphs(args):
    path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt, \
    test_set_labels_knn, num_outputs, use_continued_prediction, feature_name_map, evaluation_name, is_anomaly = args
    generate_graphs(path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction, feature_name_map, evaluation_name,
                    is_anomaly)


with open(pregen_path, 'rb') as file:
    data = pickle.load(file)
    anomaly_data = pickle.load(open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_anomaly_data.pkl"))
    with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb') as file:
        map_args = []
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

        # Anomaly data sets
        for i in range(len(data.temporary_test_set_labels_dt)):
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

        for k in range(len(test_set_names)):
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

            map_args.append([path, "evaluation_continued", model_dt, test_set_features_dt, test_set_features_knn,
                             test_set_labels_dt, test_set_labels_knn, data.num_outputs, True, data.name_map_features,
                             evaluation_name, False])

            map_args.append([path, "evaluation", model_dt, test_set_features_dt, test_set_features_knn,
                             test_set_labels_dt, test_set_labels_knn, data.num_outputs, False, data.name_map_features,
                             evaluation_name, False])

            # Wrong previous location
            test_set_features_dt_random_location = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn_random_location = np.asarray(test_sets_knn[k]).copy()

            for i in range(len(test_set_features_dt)):
                test_set_features_dt_random_location[i][0] = random.randint(0, data.num_outputs - 1)
                test_set_features_dt_random_location[i][1] = random.randint(1, data.num_outputs - 1)
                test_set_features_knn_random_location[i][0] = random.randint(0, data.num_outputs - 1) / (
                        data.num_outputs - 1)
                test_set_features_knn_random_location[i][1] = random.randint(1, data.num_outputs - 1) / (
                        data.num_outputs - 1)

            map_args.append([path, "random_prev_location", model_dt, test_set_features_dt_random_location,
                             test_set_features_knn_random_location, test_set_labels_dt, test_set_labels_knn,
                             data.num_outputs, False, data.name_map_features, evaluation_name, False])

            # Continued prediction with faulty beginning
            test_set_features_dt = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn = np.asarray(test_sets_knn[k]).copy()
            test_set_labels_dt = np.asarray(test_labels_dt[k]).copy()
            test_set_labels_knn = np.asarray(test_labels_knn[k]).copy()

            test_set_features_dt[0][0] = 5
            test_set_features_dt[0][1] = 5
            test_set_features_knn[0][0] = 5 / (data.num_outputs - 1)
            test_set_features_knn[0][1] = 5 / (data.num_outputs - 1)

            map_args.append([path, "continued_pred_with_faulty_start", model_dt, test_set_features_dt,
                             test_set_features_knn, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             True, data.name_map_features, evaluation_name, False])

            log_compiled = open(path + "evaluation/log_compiled.csv", "w")
            log_compiled.write("accuracy,accuracy_given_previous_location_was_correct,"
                               "accuracy_given_location_is_cont_the_same_and_within_5_entries,"
                               "accuracy_given_location_is_cont_the_same_and_within_10_entries,"
                               "average_path_recognition_delay,times_not_found_path\n")
            log_compiled.close()

            log_compiled = open(path + "evaluation/log_compiled_location.csv", "w")
            log_compiled.write("location,times_misclassified_as,times_misclassified,total_location\n")
            log_compiled.close()

            log_compiled = open(path + "evaluation/log_compiled_path.csv", "w")
            log_compiled.write("path_segment,recognized_after,times_misclassified,path_len\n")
            log_compiled.close()

        for i in range(len(data.temporary_test_set_labels_dt)):
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(
                data.temporary_test_set_features_dt[i],
                data.temporary_test_set_labels_dt[i],
                data.temporary_test_set_features_knn[i],
                data.temporary_test_set_labels_knn[i])

            path = BIN_FOLDER_PATH + "/" + evaluation_name + "/anomaly_model_" + data.name_map_data_sets_temporary[
                i] + "/"
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

            map_args.append(
                [path, "evaluation_continued", model_anomaly_dt, test_set_features_dt, test_set_features_knn,
                 test_set_labels_dt, test_set_labels_knn, data.num_outputs, True, data.name_map_features,
                 evaluation_name, True])

            map_args.append([path, "evaluation", model_anomaly_dt, test_set_features_dt, test_set_features_knn,
                             test_set_labels_dt, test_set_labels_knn, data.num_outputs, False, data.name_map_features,
                             evaluation_name, True])

        # Evaluate all graphs in parallel
        with Pool(NUM_CORES) as pool:
            pool.map(exec_gen_graphs, map_args)
            pool.close()
            pool.join()
