import pickle
import random
import numpy as np

from tensorflow import keras

from sources.metric.graph_feature_importance import GraphFeatureImportance
from sources.metric.graph_location_misclassified import GraphLocationMisclassified
from sources.metric.graph_location_misclassified_distribution import GraphLocationMisclassifiedDistribution
from sources.metric.graph_location_missclassification import GraphLocationMisclassification
from sources.metric.graph_path_segment_misclassified import GraphPathSegmentMisclassified
from sources.metric.graph_recognized_path_segment import GraphRecognizedPathSegment
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted


def generate_graphs(prefix, model_dt, model_knn, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs, use_continued_prediction):
    GraphTrueVsPredicted(prefix + "_dt", model_dt, True, test_set_features_dt,
                         test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphTrueVsPredicted(prefix + "_knn", model_knn, False, test_set_features_knn,
                         test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphRecognizedPathSegment(prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphRecognizedPathSegment(prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassified(prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassified(prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassification(prefix + "_dt", model_dt, True, test_set_features_dt,
                                   test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassification(prefix + "_knn", model_knn, False, test_set_features_knn,
                                   test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphLocationMisclassifiedDistribution(prefix + "_dt", model_dt, True, test_set_features_dt,
                                           test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphLocationMisclassifiedDistribution(prefix + "_knn", model_knn, False, test_set_features_knn,
                                           test_set_labels_knn, num_outputs, use_continued_prediction)

    GraphPathSegmentMisclassified(prefix + "_dt", model_dt, True, test_set_features_dt,
                                  test_set_labels_dt, num_outputs, use_continued_prediction)
    GraphPathSegmentMisclassified(prefix + "_knn", model_knn, False, test_set_features_knn,
                                  test_set_labels_knn, num_outputs, use_continued_prediction)


with open("/home/shino/Uni/master_thesis/bin/evaluation_data.pkl", 'rb') as file:
    data = pickle.load(file)
    with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'rb') as file:
        model_dt = pickle.load(file)
        model_knn = keras.models.load_model("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")

        # Feature Importance
        GraphFeatureImportance("evaluation", model_dt)

        # Valid set
        test_set_features_dt = np.asarray(data.result_features_dt[0][19]).copy()
        test_set_features_knn = np.asarray(data.result_features_knn[0][19]).copy()
        test_set_labels_dt = np.asarray(data.result_labels_dt[0][19]).copy()
        test_set_labels_knn = np.asarray(data.result_labels_knn[0][19]).copy()

        generate_graphs("evaluation_continued", model_dt, model_knn, test_set_features_dt, test_set_features_knn,
                        test_set_labels_dt, test_set_labels_knn, data.num_outputs, True)

        generate_graphs("evaluation", model_dt, model_knn, test_set_features_dt, test_set_features_knn,
                        test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Previous Location with offset
        test_set_features_dt_offset = np.asarray(data.result_features_dt[0][19]).copy()
        test_set_features_knn_offset = np.asarray(data.result_features_knn[0][19]).copy()

        for i in range(10, len(test_set_features_dt)):
            test_set_features_dt_offset[i][0] = test_set_features_dt_offset[i - 10][0]
            test_set_features_dt_offset[i][1] = test_set_features_dt_offset[i - 10][1]
            test_set_features_knn_offset[i][0] = test_set_features_knn_offset[i - 10][0]
            test_set_features_knn_offset[i][1] = test_set_features_knn_offset[i - 10][1]

        generate_graphs("prev_location_offset", model_dt, model_knn, test_set_features_dt_offset,
                        test_set_features_knn_offset, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Wrong previous location
        test_set_features_dt_random_location = np.asarray(data.result_features_dt[0][19]).copy()
        test_set_features_knn_random_location = np.asarray(data.result_features_knn[0][19]).copy()

        for i in range(len(test_set_features_dt)):
            test_set_features_dt_random_location[i][0] = random.randint(0, data.num_outputs)
            test_set_features_dt_random_location[i][1] = random.randint(1, data.num_outputs)
            test_set_features_knn_random_location[i][0] = random.randint(0, data.num_outputs) / data.num_outputs
            test_set_features_knn_random_location[i][1] = random.randint(1, data.num_outputs) / data.num_outputs

        generate_graphs("random_prev_location", model_dt, model_knn, test_set_features_dt_random_location,
                        test_set_features_knn_random_location, test_set_labels_dt, test_set_labels_knn,
                        data.num_outputs, False)

        # Nulled Acceleraton
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[1][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[1][19]).copy()

        generate_graphs("nulled_acceleration", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Nulled Light
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[2][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[2][19]).copy()

        generate_graphs("nulled_light", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Nulled Accesspoint
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[3][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[3][19]).copy()

        generate_graphs("nulled_access_point", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Nulled Temperature
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[5][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[5][19]).copy()

        generate_graphs("nulled_temperature", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Nulled Heading
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[4][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[4][19]).copy()

        generate_graphs("nulled_heading", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Nulled Volume
        test_set_features_dt_nulled = np.asarray(data.faulty_features_dt[6][19]).copy()
        test_set_features_knn_nulled = np.asarray(data.faulty_features_knn[6][19]).copy()

        generate_graphs("nulled_volume", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs, False)

        # Permuted path
        test_set_features_dt_faulty = np.asarray(data.faulty_features_dt[0][18]).copy()
        test_set_features_knn_faulty = np.asarray(data.faulty_features_knn[0][18]).copy()
        test_set_labels_dt_faulty = np.asarray(data.faulty_labels_dt[0][18]).copy()
        test_set_labels_knn_faulty = np.asarray(data.faulty_labels_knn[0][18]).copy()

        generate_graphs("permuted_path", model_dt, model_knn, test_set_features_dt_faulty,
                        test_set_features_knn_faulty, test_set_labels_dt_faulty, test_set_labels_knn_faulty,
                        data.num_outputs, False)

        # Continued prediction with faulty beginning
        test_set_features_dt = np.asarray(data.result_features_dt[0][19]).copy()
        test_set_features_knn = np.asarray(data.result_features_knn[0][19]).copy()
        test_set_labels_dt = np.asarray(data.result_labels_dt[0][19]).copy()
        test_set_labels_knn = np.asarray(data.result_labels_knn[0][19]).copy()

        test_set_features_dt[0][0] = 5
        test_set_features_dt[0][1] = 5
        test_set_features_knn[0][0] = 5
        test_set_features_knn[0][1] = 5

        generate_graphs("continued_pred_with_faulty_start", model_dt, model_knn, test_set_features_dt, test_set_features_knn,
                        test_set_labels_dt, test_set_labels_knn, data.num_outputs, True)
