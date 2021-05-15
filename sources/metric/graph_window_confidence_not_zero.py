import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class GraphWindowConfidenceNotZero:
    def __init__(self, path, prefix, prediction, is_dt, encode_paths):
        self.prefix = prefix
        self.prediction = prediction
        self.is_dt = is_dt
        self.encode_paths = encode_paths

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
        return "{0}_window_confidence_not_zero.png".format(self.prefix)

    def __generate_graph(self):
        location_changes_in_window = self.__get_window_confidence_data(self.prediction)

        fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
        plt.plot(range(min(self.max_entry_draw_size, len(location_changes_in_window))), location_changes_in_window[:self.max_entry_draw_size], "o-b")
        plt.xlabel("Pfadeintrag (Diskret)")
        plt.ylabel("Akkumulierte Standortwahrscheinlichkeit im Fenster")
        # plt.title("Akkumulierte Positionswahrscheinlichkeit im Fenster ({0}) ohne 0-Ort".format(self.window_size))
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)

    def __get_window_confidence_data(self, predicted):
        window_confidence = []
        confidence = []
        for pred_label in predicted:
            new_location = self.__get_discrete_label(pred_label)
            if new_location > 0:
                confidence.append(pred_label[new_location])

                if len(confidence) > self.window_size:
                    confidence.pop(0)
                window_confidence.append(sum(confidence))
        return window_confidence

    def __get_discrete_label(self, label):
        if self.is_dt and self.encode_paths:
            return np.asarray(label).argmax() + 1
        return np.asarray(label).argmax()
