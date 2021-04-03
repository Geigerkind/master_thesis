import pickle

from tensorflow import keras

from sources.metric.graph_location_misclassified import GraphLocationMisclassified
from sources.metric.graph_recognized_path_segment import GraphRecognizedPathSegment
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted

with open("/home/shino/Uni/master_thesis/bin/evaluation_data.pkl", 'rb') as file:
    data = pickle.load(file)
    with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'rb') as file:
        model_dt = pickle.load(file)
        model_knn = keras.models.load_model("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")

        GraphTrueVsPredicted("evaluation_dt", model_dt, True, data.result_features_dt[0][19],
                             data.result_labels_dt[0][19], data.num_outputs)
        GraphTrueVsPredicted("evaluation_knn", model_knn, False, data.result_features_knn[0][19],
                             data.result_labels_knn[0][19], data.num_outputs)

        GraphRecognizedPathSegment("evaluation_dt", model_dt, True, data.result_features_dt[0][19],
                                   data.result_labels_dt[0][19])
        GraphRecognizedPathSegment("evaluation_knn", model_knn, False, data.result_features_knn[0][19],
                                   data.result_labels_knn[0][19])

        GraphLocationMisclassified("evaluation_dt", model_dt, True, data.result_features_dt[0][19],
                                   data.result_labels_dt[0][19], data.num_outputs)
        GraphLocationMisclassified("evaluation_knn", model_knn, False, data.result_features_knn[0][19],
                                   data.result_labels_knn[0][19], data.num_outputs)
