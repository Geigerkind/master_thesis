import os

import matplotlib.pyplot as plt
import numpy as np


class GraphWindowConfidence():
    def __init__(self, path, prefix, prediction):
        self.prefix = prefix
        self.prediction = prediction

        # Configuration
        self.max_entry_draw_size = 2000
        self.window_size = 25
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_window_confidence.png".format(self.prefix)

    def __generate_graph(self):
        location_changes_in_window = self.__get_window_confidence_data(self.prediction)

        plt.figure(figsize=(15/2.54, 30/2.54))
        fig, ax1 = plt.subplots()
        ax1.plot(range(min(self.max_entry_draw_size, len(location_changes_in_window))), location_changes_in_window[:self.max_entry_draw_size], "o-b")
        ax1.set_xlabel("Pfadeintrag (Diskret)")
        ax1.set_ylabel("Akkumulierte Positionswahrscheinlichkeit im Fenster")
        ax1.set_title("Akkumulierte Positionswahrscheinlichkeit im Fenster ({0})".format(self.window_size))
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)

    def __get_window_confidence_data(self, predicted):
        window_confidence = []
        confidence = []
        for pred_label in predicted:
            new_location = self.__get_discrete_label(pred_label)
            confidence.append(pred_label[new_location])

            if len(confidence) > self.window_size:
                confidence.pop(0)
            window_confidence.append(sum(confidence))
        return window_confidence

    def __get_discrete_label(self, label):
        return np.asarray(label).argmax()
