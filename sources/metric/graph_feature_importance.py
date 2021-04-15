import os

import matplotlib.pyplot as plt
import numpy as np


class GraphFeatureImportance:
    def __init__(self, prefix, model_dt):
        self.prefix = prefix
        self.model_dt = model_dt

        # Configuration
        self.file_path = "/home/shino/Uni/master_thesis/bin/main_evaluation/" + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_feature_importance.png".format(self.prefix)

    def __generate_graph(self):
        importances = self.model_dt.feature_importances()

        fig, ax1 = plt.subplots()
        ax1.bar(range(len(importances)), importances)
        ax1.set_xlabel("Feature (Diskret)")
        ax1.set_ylabel("Anteil")
        ax1.set_ylim([0, min(1, 2 * np.asarray(importances).max())])
        ax1.set_title("Verteilung der Nutzung der Features")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)
