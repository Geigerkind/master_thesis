import os

import matplotlib.pyplot as plt
import numpy as np


class GraphWindowLocationChanges():
    def __init__(self, path, prefix, prediction):
        self.prefix = prefix
        self.prediction = prediction

        # Configuration
        self.window_size = 25
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_window_location_changes.png".format(self.prefix)

    def __generate_graph(self):
        location_changes_in_window = self.__get_window_location_change_frequency_data(self.prediction)

        fig, ax1 = plt.subplots()
        ax1.plot(range(len(location_changes_in_window)), location_changes_in_window, "o-b")
        ax1.set_xlabel("Pfadeintrag (Diskret)")
        ax1.set_ylabel("Positionsänderungen im Fenster")
        ax1.set_title("Positionsänderungen im Fenster ({0})".format(self.window_size))
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)

    def __get_window_location_change_frequency_data(self, predicted):
        window_location_changes = []
        location_changes = []
        current_location = self.__get_discrete_label(predicted[0])
        for pred_label in predicted:
            new_location = self.__get_discrete_label(pred_label)
            if new_location != current_location:
                current_location = new_location
                location_changes.append(1)
            else:
                location_changes.append(0)

            if len(location_changes) > self.window_size:
                location_changes.pop(0)
            window_location_changes.append(sum(location_changes))
        return window_location_changes

    def __get_discrete_label(self, label):
        return np.asarray(label).argmax()
