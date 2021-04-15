import os
import pickle
import random
from multiprocessing import cpu_count, Pool

import numpy as np
from tensorflow import keras

from sources.metric.compile_log import CompileLog
from sources.metric.graph_feature_importance import GraphFeatureImportance
from sources.metric.graph_location_misclassified import GraphLocationMisclassified
from sources.metric.graph_location_misclassified_distribution import GraphLocationMisclassifiedDistribution
from sources.metric.graph_location_missclassification import GraphLocationMisclassification
from sources.metric.graph_path_segment_misclassified import GraphPathSegmentMisclassified
from sources.metric.graph_recognized_path_segment import GraphRecognizedPathSegment
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted


def generate_graphs(path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction):
    # Loaded here cause it cant be pickled
    model_knn = keras.models.load_model("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")
    GraphTrueVsPredicted(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                         test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphTrueVsPredicted(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                         test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphRecognizedPathSegment(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphRecognizedPathSegment(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassified(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassified(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassification(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                                   test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassification(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                                   test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassifiedDistribution(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                                           test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassifiedDistribution(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                                           test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphPathSegmentMisclassified(path, prefix + "_dt", model_dt, True, test_set_features_dt,
                                  test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphPathSegmentMisclassified(path, prefix + "_knn", model_knn, False, test_set_features_knn,
                                  test_set_labels_knn, num_outputs, use_continued_prediction)

    CompileLog(path, prefix + "_dt")
    CompileLog(path, prefix + "_knn")


def exec_gen_graphs(args):
    path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt, \
    test_set_labels_knn, num_outputs, use_continued_prediction = args
    generate_graphs(path, prefix, model_dt, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction)


with open("/home/shino/Uni/master_thesis/bin/evaluation_data.pkl", 'rb') as file:
    data = pickle.load(file)
    with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'rb') as file:
        map_args = []
        model_dt = pickle.load(file)

        try:
            os.mkdir("/home/shino/Uni/master_thesis/bin/main_evaluation/")
        except:
            pass

        # Feature Importance
        GraphFeatureImportance("evaluation", model_dt)


        def glue_test_sets(features_dt, labels_dt, features_knn, labels_knn):
            new_set_dt = []
            new_labels_dt = []
            new_set_knn = []
            new_labels_knn = []
            for j in range(5):
                new_set_dt = new_set_dt + features_dt[15 + j]
                new_labels_dt = new_labels_dt + labels_dt[15 + j]
                new_set_knn = new_set_knn + features_knn[15 + j]
                new_labels_knn = new_labels_knn + labels_knn[15 + j]

            return new_set_dt, new_labels_dt, new_set_knn, new_labels_knn


        test_sets_dt = []
        test_labels_dt = []
        test_sets_knn = []
        test_labels_knn = []
        for i in range(4):
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(data.result_features_dt[i],
                                                                                    data.result_labels_dt[i],
                                                                                    data.result_features_knn[i],
                                                                                    data.result_labels_knn[i])

            test_sets_dt.append(np.asarray(new_set_dt).copy())
            test_labels_dt.append(np.asarray(new_labels_dt).copy())
            test_sets_knn.append(np.asarray(new_set_knn).copy())
            test_labels_knn.append(np.asarray(new_labels_knn).copy())

        new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(data.test_route_features_dt[0],
                                                                                data.test_route_labels_dt[0],
                                                                                data.test_route_features_knn[0],
                                                                                data.test_route_labels_knn[0])
        test_sets_dt.append(np.asarray(new_set_dt).copy())
        test_labels_dt.append(np.asarray(new_labels_dt).copy())
        test_sets_knn.append(np.asarray(new_set_knn).copy())
        test_labels_knn.append(np.asarray(new_labels_knn).copy())

        for i in range(7):
            new_set_dt, new_labels_dt, new_set_knn, new_labels_knn = glue_test_sets(data.faulty_features_dt[28 + i],
                                                                                    data.faulty_labels_dt[28 + i],
                                                                                    data.faulty_features_knn[28 + i],
                                                                                    data.faulty_labels_knn[28 + i])

            test_sets_dt.append(np.asarray(new_set_dt).copy())
            test_labels_dt.append(np.asarray(new_labels_dt).copy())
            test_sets_knn.append(np.asarray(new_set_knn).copy())
            test_labels_knn.append(np.asarray(new_labels_knn).copy())

        test_set_names = ["simple_square", "long_rectangle", "rectangle_with_ramp", "many_corners", "combination",
                          "faulty_permuted_paths", "faulty_nulled_acceleration", "faulty_nulled_light",
                          "faulty_nulled_access_point", "faulty_nulled_heading", "faulty_nulled_temperature",
                          "faulty_nulled_voluime"]
        for k in range(len(test_set_names)):
            path = "/home/shino/Uni/master_thesis/bin/main_evaluation/" + test_set_names[k] + "/"
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
                             test_set_labels_dt, test_set_labels_knn, data.num_outputs, True])

            map_args.append([path, "evaluation", model_dt, test_set_features_dt, test_set_features_knn,
                             test_set_labels_dt, test_set_labels_knn, data.num_outputs, False])

            # Previous Location with offset
            test_set_features_dt_offset = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn_offset = np.asarray(test_sets_knn[k]).copy()

            for i in range(10, len(test_set_features_dt)):
                test_set_features_dt_offset[i][0] = test_set_features_dt_offset[i - 10][0]
                test_set_features_dt_offset[i][1] = test_set_features_dt_offset[i - 10][1]
                test_set_features_knn_offset[i][0] = test_set_features_knn_offset[i - 10][0]
                test_set_features_knn_offset[i][1] = test_set_features_knn_offset[i - 10][1]

            map_args.append([path, "prev_location_offset", model_dt, test_set_features_dt_offset,
                             test_set_features_knn_offset, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Wrong previous location
            test_set_features_dt_random_location = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn_random_location = np.asarray(test_sets_knn[k]).copy()

            for i in range(len(test_set_features_dt)):
                test_set_features_dt_random_location[i][0] = random.randint(0, data.num_outputs)
                test_set_features_dt_random_location[i][1] = random.randint(1, data.num_outputs)
                test_set_features_knn_random_location[i][0] = random.randint(0, data.num_outputs) / data.num_outputs
                test_set_features_knn_random_location[i][1] = random.randint(1, data.num_outputs) / data.num_outputs

            map_args.append([path, "random_prev_location", model_dt, test_set_features_dt_random_location,
                             test_set_features_knn_random_location, test_set_labels_dt, test_set_labels_knn,
                             data.num_outputs, False])

            # Continued prediction with faulty beginning
            test_set_features_dt = np.asarray(test_sets_dt[k]).copy()
            test_set_features_knn = np.asarray(test_sets_knn[k]).copy()
            test_set_labels_dt = np.asarray(test_labels_dt[k]).copy()
            test_set_labels_knn = np.asarray(test_labels_knn[k]).copy()

            test_set_features_dt[0][0] = 5
            test_set_features_dt[0][1] = 5
            test_set_features_knn[0][0] = 5 / data.num_outputs
            test_set_features_knn[0][1] = 5 / data.num_outputs

            map_args.append([path, "continued_pred_with_faulty_start", model_dt, test_set_features_dt,
                             test_set_features_knn, test_set_labels_dt, test_set_labels_knn, data.num_outputs, True])

            log_compiled = open("/home/shino/Uni/master_thesis/bin/main_evaluation/evaluation/log_compiled.csv", "w")
            log_compiled.write("accuracy,accuracy_given_previous_location_was_correct,"
                               "accuracy_given_location_is_cont_the_same_and_within_5_entries,"
                               "accuracy_given_location_is_cont_the_same_and_within_10_entries,"
                               "average_path_recognition_delay,times_not_found_path\n")
            log_compiled.close()

            log_compiled = open(
                "/home/shino/Uni/master_thesis/bin/main_evaluation/evaluation/log_compiled_location.csv", "w")
            log_compiled.write("location,times_misclassified_as,times_misclassified,total_location\n")
            log_compiled.close()

            log_compiled = open("/home/shino/Uni/master_thesis/bin/main_evaluation/evaluation/log_compiled_path.csv",
                                "w")
            log_compiled.write("path_segment,recognized_after,times_misclassified,path_len\n")
            log_compiled.close()

        # Evaluate all graphs in parallel
        with Pool(cpu_count()) as pool:
            pool.map(exec_gen_graphs, map_args)
            pool.close()
            pool.join()

# TODO
"""
# Nulled Acceleraton
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[1][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[1][19]).copy()

            map_args.append(["nulled_acceleration", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Nulled Light
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[2][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[2][19]).copy()

            map_args.append(["nulled_light", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Nulled Accesspoint
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[3][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[3][19]).copy()

            map_args.append(["nulled_access_point", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Nulled Temperature
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[5][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[5][19]).copy()

            map_args.append(["nulled_temperature", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Nulled Heading
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[4][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[4][19]).copy()

            map_args.append(["nulled_heading", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Nulled Volume
            test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[6][19]).copy()
            test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[6][19]).copy()

            map_args.append(["nulled_volume", model_dt, test_set_features_dt_nulled,
                             test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs,
                             False])

            # Permuted path
            test_set_features_dt_faulty = np.asarray(data.faulty_features_dt[0][18]).copy()
            test_set_features_knn_faulty = np.asarray(data.faulty_features_knn[0][18]).copy()
            test_set_labels_dt_faulty = np.asarray(data.faulty_labels_dt[0][18]).copy()
            test_set_labels_knn_faulty = np.asarray(data.faulty_labels_knn[0][18]).copy()

            map_args.append(["permuted_path", model_dt, test_set_features_dt_faulty,
                             test_set_features_knn_faulty, test_set_labels_dt_faulty, test_set_labels_knn_faulty,
                             data.num_outputs, False])
"""
