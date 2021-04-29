import os

import matplotlib.pyplot as plt
import numpy as np


class GraphLocationMisclassified:
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
        return "{0}_location_misclassified.png".format(self.prefix)

    def __get_discrete_label(self, label):
        if self.is_dt:
            return label
        return np.asarray(label).argmax()

    def __compare_predicted(self, test, predicted):
        if self.is_dt:
            return test == predicted
        return self.__get_discrete_label(test) == self.__get_discrete_label(predicted)

    def __calculate_location_misclassified(self):
        result = dict()
        total = dict()
        for i in range(int(self.num_outputs)):
            result[i] = 0
            total[i] = 0

        for i in range(len(self.test_labels)):
            real_label = self.__get_discrete_label(self.test_labels[i])
            total[real_label] = total[real_label] + 1
            if not self.__compare_predicted(self.test_labels[i], self.prediction[i]):
                result[real_label] = result[real_label] + 1

        log_file = open(self.file_path + "log_location_misclassified.csv", "w")
        log_file.write("location,times_misclassified,total_location\n")
        keys = []
        values = []
        for key in result:
            log_file.write("{0},{1},{2}\n".format(key, result[key], total[key]))
            keys.append(key)
            if total[key] == 0:
                values.append(0)
            else:
                values.append(result[key] / total[key])
        log_file.close()
        return keys, values

    def __generate_graph(self):
        plt.figure(figsize=(15/2.54, 30/2.54))
        x, y = self.__calculate_location_misclassified()
        plt.bar(x, y)
        plt.xlabel("Ort (Diskret)")
        plt.ylabel("Anteil falsch klassifiziert")
        plt.ylim([0, 1])
        plt.title("Anteil Orte falsch klassifiziert")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
