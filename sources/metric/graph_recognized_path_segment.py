import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class GraphRecognizedPathSegment:
    def __init__(self, path, prefix, is_dt, test_labels, prediction):
        self.prefix = prefix
        self.is_dt = is_dt
        self.test_labels = np.asarray(test_labels)
        self.prediction = prediction

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass
        self.infinite_number = 99999999

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_recognized_path_segment.png".format(self.prefix)

    def __get_discrete_label(self, label):
        if self.is_dt:
            return label
        return np.asarray(label).argmax()

    def __compare_predicted(self, test, predicted):
        if self.is_dt:
            return test == predicted
        return self.__get_discrete_label(test) == self.__get_discrete_label(predicted)

    def __calculate_path_segments(self):
        result = []

        current_location = self.__get_discrete_label(self.test_labels[0])
        previous_prediction = self.__get_discrete_label(self.prediction[0])
        current_location_guessed = False
        count = 0
        same_prediction_count = 1
        for i in range(len(self.test_labels)):
            if current_location != self.__get_discrete_label(self.test_labels[i]):
                if not current_location_guessed:
                    result.append(self.infinite_number)
                current_location = self.__get_discrete_label(self.test_labels[i])
                current_location_guessed = False
                count = 0

                if current_location == previous_prediction:
                    current_location_guessed = True
                    result.append(-same_prediction_count)
                    same_prediction_count = 0

            if self.__compare_predicted(self.test_labels[i], self.prediction[i]):
                if not current_location_guessed:
                    result.append(count)
                    current_location_guessed = True
                same_prediction_count = same_prediction_count + 1
            else:
                count = count + 1
                same_prediction_count = 1
            previous_prediction = self.__get_discrete_label(self.prediction[i])

        return np.asarray(result)

    def __generate_graph(self):
        fig = plt.figure(figsize=(30/2.54, 15/2.54))
        path_segments = self.__calculate_path_segments()

        log_file = open(self.file_path + "log_recognized_path_segment.csv", "w")
        log_file.write("path_segment,recognized_after\n")
        for i in range(len(path_segments)):
            log_file.write("{0},{1}\n".format(i, path_segments[i]))
        log_file.close()

        min_value = path_segments.min()
        filtered_path_segments = path_segments[path_segments != self.infinite_number]
        max_value = 1 if len(filtered_path_segments) == 0 else filtered_path_segments.max() + 1
        plt.bar(range(len(path_segments)), path_segments)
        for i in range(len(path_segments)):
            if path_segments[i] == self.infinite_number:
                plt.text(i, (max_value - min_value) // 2, "INF", ha="center")

        plt.xlabel("Pfadabschnitt (Diskret)")
        plt.ylabel("Erkennung nach Anzahl Pfadeintr??gen (Diskret)")
        plt.ylim([min_value, max_value])
        # plt.title("Pfadabschnitterkennung")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)
