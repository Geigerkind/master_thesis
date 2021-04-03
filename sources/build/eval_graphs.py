import pickle
import random

from tensorflow import keras

from sources.metric.graph_location_misclassified import GraphLocationMisclassified
from sources.metric.graph_location_misclassified_distribution import GraphLocationMisclassifiedDistribution
from sources.metric.graph_location_missclassification import GraphLocationMisclassification
from sources.metric.graph_path_segment_misclassified import GraphPathSegmentMisclassified
from sources.metric.graph_recognized_path_segment import GraphRecognizedPathSegment
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted


def generate_graphs(prefix, model_dt, model_knn, test_set_features_dt, test_set_features_knn, test_set_labels_dt,
                    test_set_labels_knn, num_outputs):
    GraphTrueVsPredicted(prefix + "_dt", model_dt, True, test_set_features_dt,
                         test_set_labels_dt, num_outputs)
    GraphTrueVsPredicted(prefix + "_knn", model_knn, False, test_set_features_knn,
                         test_set_labels_knn, num_outputs)

    GraphRecognizedPathSegment(prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt)
    GraphRecognizedPathSegment(prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn)

    GraphLocationMisclassified(prefix + "_dt", model_dt, True, test_set_features_dt,
                               test_set_labels_dt, num_outputs)
    GraphLocationMisclassified(prefix + "_knn", model_knn, False, test_set_features_knn,
                               test_set_labels_knn, num_outputs)

    GraphLocationMisclassification(prefix + "_dt", model_dt, True, test_set_features_dt,
                                   test_set_labels_dt, num_outputs)
    GraphLocationMisclassification(prefix + "_knn", model_knn, False, test_set_features_knn,
                                   test_set_labels_knn, num_outputs)

    GraphLocationMisclassifiedDistribution(prefix + "_dt", model_dt, True, test_set_features_dt,
                                           test_set_labels_dt, num_outputs)
    GraphLocationMisclassifiedDistribution(prefix + "_knn", model_knn, False, test_set_features_knn,
                                           test_set_labels_knn, num_outputs)

    GraphPathSegmentMisclassified(prefix + "_dt", model_dt, True, test_set_features_dt,
                                  test_set_labels_dt)
    GraphPathSegmentMisclassified(prefix + "_knn", model_knn, False, test_set_features_knn,
                                  test_set_labels_knn)


with open("/home/shino/Uni/master_thesis/bin/evaluation_data.pkl", 'rb') as file:
    data = pickle.load(file)
    with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'rb') as file:
        model_dt = pickle.load(file)
        model_knn = keras.models.load_model("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")

        # Valid set
        test_set_features_dt = data.result_features_dt[0][19]
        test_set_features_knn = data.result_features_knn[0][19]
        test_set_labels_dt = data.result_labels_dt[0][19]
        test_set_labels_knn = data.result_labels_knn[0][19]

        generate_graphs("evaluation", model_dt, model_knn, test_set_features_dt, test_set_features_knn,
                        test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Previous Location with offset
        test_set_features_dt_offset = data.result_features_dt[0][19]
        test_set_features_knn_offset = data.result_features_knn[0][19]

        for i in range(10, len(test_set_features_dt)):
            test_set_features_dt_offset[i][0] = test_set_features_dt_offset[i - 10][0]
            test_set_features_dt_offset[i][1] = test_set_features_dt_offset[i - 10][1]
            test_set_features_knn_offset[i][0] = test_set_features_knn_offset[i - 10][0]
            test_set_features_knn_offset[i][1] = test_set_features_knn_offset[i - 10][1]

        generate_graphs("prev_location_offset", model_dt, model_knn, test_set_features_dt_offset,
                        test_set_features_knn_offset, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Wrong previous location
        test_set_features_dt_random_location = data.result_features_dt[0][19]
        test_set_features_knn_random_location = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            test_set_features_dt_random_location[i][0] = random.randint(0, data.num_outputs)
            test_set_features_dt_random_location[i][1] = random.randint(1, data.num_outputs)
            test_set_features_knn_random_location[i][0] = random.randint(0, data.num_outputs) / data.num_outputs
            test_set_features_knn_random_location[i][1] = random.randint(1, data.num_outputs) / data.num_outputs

        generate_graphs("random_prev_location", model_dt, model_knn, test_set_features_dt_random_location,
                        test_set_features_knn_random_location, test_set_labels_dt, test_set_labels_knn,
                        data.num_outputs)

        # Nulled Acceleraton
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(2, 6):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_acceleration", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Nulled Light
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(6, 10):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_light", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Nulled Accesspoint
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(10, 15):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_access_point", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Nulled Temperature
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(15, 19):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_temperature", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Nulled Heading
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(19, 23):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_heading", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Nulled Volume
        test_set_features_dt_nulled = data.result_features_dt[0][19]
        test_set_features_knn_nulled = data.result_features_knn[0][19]

        for i in range(len(test_set_features_dt)):
            for j in range(23, 27):
                test_set_features_dt_nulled[i][j] = 0
                test_set_features_knn_nulled[i][j] = 0

        generate_graphs("nulled_volume", model_dt, model_knn, test_set_features_dt_nulled,
                        test_set_features_knn_nulled, test_set_labels_dt, test_set_labels_knn, data.num_outputs)

        # Faulty data set
        test_set_features_dt_faulty = data.faulty_features_dt[0][18]
        test_set_features_knn_faulty = data.faulty_features_knn[0][18]
        test_set_labels_dt_faulty = data.faulty_labels_dt[0][18]
        test_set_labels_knn_faulty = data.faulty_labels_knn[0][18]

        generate_graphs("faulty", model_dt, model_knn, test_set_features_dt_faulty,
                        test_set_features_knn_faulty, test_set_labels_dt_faulty, test_set_labels_knn_faulty,
                        data.num_outputs)
