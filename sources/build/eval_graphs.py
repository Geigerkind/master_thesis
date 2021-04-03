import pickle

from tensorflow import keras

from sources.data.data_compiler import DataCompiler
from sources.data.data_set import DataSet
from sources.data.features import Features
from sources.metric.graph_true_vs_predicted import GraphTrueVsPredicted

features = [Features.PreviousLocation, Features.AccessPointDetection, Features.Temperature, Features.StandardDeviation]
data = DataCompiler([DataSet.SimpleSquare], features, False)
# data = DataCompiler([DataSet.SimpleSquare, DataSet.LongRectangle, DataSet.RectangleWithRamp, DataSet.ManyCorners], features)

with open("/home/shino/Uni/master_thesis/bin/evaluation_dt_model.pkl", 'rb') as file:
    model_dt = pickle.load(file)
    model_knn = keras.models.load_model("/home/shino/Uni/master_thesis/bin/evaluation_knn_model.h5")

    GraphTrueVsPredicted("evaluation_dt", model_dt, True, data.result_features_dt[0][19], data.result_labels_dt[0][19],
                         data.num_outputs)
    GraphTrueVsPredicted("evaluation_knn", model_knn, False, data.result_features_knn[0][19],
                         data.result_labels_knn[0][19], data.num_outputs)
