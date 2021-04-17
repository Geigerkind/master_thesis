import os

import matplotlib.pyplot as plt
import numpy as np


class GraphLocationMisclassification:
    def __init__(self, path, prefix, is_dt, test_labels, num_outputs, prediction):
        self.prefix = prefix
        self.is_dt = is_dt
        self.test_labels = np.asarray(test_labels)
        self.num_outputs = num_outputs
        self.prediction = prediction

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_location_misclassification.png".format(self.prefix)

    def __get_discrete_label(self, label):
        if self.is_dt:
            return label
        return np.asarray(label).argmax()

    def __compare_predicted(self, test, predicted):
        if self.is_dt:
            return test == predicted
        return self.__get_discrete_label(test) == self.__get_discrete_label(predicted)

    def __calculate_location_misclassification(self):
        result = dict()
        for i in range(int(self.num_outputs)):
            result[i] = 0

        total = 0
        for i in range(len(self.test_labels)):
            predicted_label = self.__get_discrete_label(self.prediction[i])
            if not self.__compare_predicted(self.test_labels[i], self.prediction[i]):
                result[predicted_label] = result[predicted_label] + 1
                total = total + 1

        log_file = open(self.file_path + "log_location_misclassification.csv", "w")
        log_file.write("location,times_misclassified_as\n")
        keys = []
        values = []
        for key in result:
            log_file.write("{0},{1}\n".format(key, result[key]))
            keys.append(key)
            if total == 0:
                values.append(0)
            else:
                values.append(result[key] / total)
        log_file.close()
        return keys, values

    def __generate_graph(self):
        x, y = self.__calculate_location_misclassification()
        plt.bar(x, y)
        plt.xlabel("Ort (Diskret)")
        plt.ylabel("Anteil Klassifizierung")
        plt.ylim([0, 1])
        plt.title("Verteilung von als was etwas missklassifiziert wurde")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
