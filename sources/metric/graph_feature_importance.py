import os

import matplotlib.pyplot as plt
import numpy as np

from sources.ffnn.gen_ffnn import GenerateFFNN


class GraphFeatureImportance:
    def __init__(self, path, prefix, model_dt, model_knn, test_features_knn, test_labels_knn, test_features_dt,
                 test_labels_dt, feature_name_map):
        self.prefix = prefix
        self.model_dt = model_dt
        self.model_knn = model_knn
        self.test_features_knn = test_features_knn
        self.test_labels_knn = test_labels_knn
        self.test_features_dt = test_features_dt
        self.test_labels_dt = test_labels_dt
        self.path = path
        self.feature_name_map = feature_name_map

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph(self.model_dt.feature_importances(), "dt")
        self.__generate_graph(self.model_dt.permutation_importance(self.test_features_dt, self.test_labels_dt), "dt_pi")
        self.__generate_graph(
            GenerateFFNN.feature_importances(self.model_knn, self.test_features_knn, self.test_labels_knn),
            "knn_pi")

    def __graph_name(self, suffix):
        return "{0}_feature_importance_{1}.png".format(self.prefix, suffix)

    def __generate_graph(self, importances, suffix):
        plt.figure(figsize=(15/2.54, 30/2.54))
        fig, ax1 = plt.subplots()
        ax1.bar(range(len(importances)), importances, align='center')
        plt.xticks(range(len(importances)), self.feature_name_map, size='small', rotation=90)
        ax1.set_xlabel("Feature (Diskret)")
        if suffix == "dt":
            ax1.set_ylabel("Anteil")
        else:
            ax1.set_ylabel("Klassifizierungsfehler")
        ax1.set_ylim([0, min(1, 2 * np.asarray(importances).max())])
        ax1.set_title("Wichtigkeit der Features")
        fig.tight_layout()
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name(suffix)))
        plt.clf()
        plt.close(fig)
