import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sources.ffnn.gen_ffnn import GenerateFFNN


class GraphFeatureImportance:
    def __init__(self, path, prefix, model_dt, model_knn, test_features_knn, test_labels_knn, test_features_dt,
                 test_labels_dt, feature_name_map, is_anamoly):
        self.prefix = prefix
        self.model_dt = model_dt
        self.model_knn = model_knn
        self.test_features_knn = test_features_knn
        self.test_labels_knn = test_labels_knn
        self.test_features_dt = test_features_dt
        self.test_labels_dt = test_labels_dt
        self.path = path
        self.feature_name_map = feature_name_map
        self.is_anamoly = is_anamoly

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph(self.model_dt.feature_importances(), "dt")
        self.__generate_graph(self.model_dt.permutation_importance(self.test_features_dt, self.test_labels_dt), "dt_pi")
        self.__generate_graph(
            GenerateFFNN.feature_importances(self.model_knn, self.test_features_knn, self.test_labels_knn, self.is_anamoly),
            "knn_pi")

    def __graph_name(self, suffix):
        return "{0}_feature_importance_{1}.png".format(self.prefix, suffix)

    def __generate_graph(self, importances, suffix):
        fig = plt.figure(figsize=(15/2.54, 10/2.54))
        plt.bar(range(len(importances)), importances, align='center')
        plt.xticks(range(len(importances)), self.feature_name_map, size='small', rotation=90)
        plt.xlabel("Feature (Diskret)")
        if suffix == "dt":
            plt.ylabel("Anteil")
        else:
            plt.ylabel("Klassifizierungsfehler")
        plt.ylim([0, min(1, 2 * np.asarray(importances).max())])
        # plt.title("Wichtigkeit der Features")
        plt.tight_layout()
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name(suffix)))
        plt.clf()
        plt.close(fig)
