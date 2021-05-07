import os

import matplotlib.pyplot as plt
import numpy as np


class GraphPathSegmentMisclassified:
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

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_path_segment_misclassified.png".format(self.prefix)

    def __get_discrete_label(self, label):
        if self.is_dt:
            return label
        return np.asarray(label).argmax()

    def __compare_predicted(self, test, predicted):
        if self.is_dt:
            return test == predicted
        return self.__get_discrete_label(test) == self.__get_discrete_label(predicted)

    def __calculate_path_segment_misclassified(self):
        result = []
        total = []
        current_location = self.__get_discrete_label(self.test_labels[0])
        segment_count = 0
        misclassified_count = 0
        for i in range(len(self.test_labels)):
            real_label = self.__get_discrete_label(self.test_labels[i])
            if real_label != current_location:
                current_location = real_label
                result.append(misclassified_count)
                total.append(segment_count)
                segment_count = 0
                misclassified_count = 0

            segment_count = segment_count + 1

            if self.__get_discrete_label(self.prediction[i]) != current_location:
                misclassified_count = misclassified_count + 1

        result.append(misclassified_count)
        total.append(segment_count)

        log_file = open(self.file_path + "log_path_segment_misclassified.csv", "w")
        log_file.write("path_segment,times_misclassified,path_len\n")
        values = []
        for i in range(len(result)):
            log_file.write("{0},{1},{2}\n".format(i, result[i], total[i]))
            if total[i] == 0:
                values.append(0)
            else:
                values.append(result[i] / total[i])
        log_file.close()

        return values

    def __generate_graph(self):
        fig = plt.figure(figsize=(30/2.54, 15/2.54))
        y = self.__calculate_path_segment_misclassified()
        plt.bar(range(len(y)), y)
        plt.xlabel("Pfadsegment (Diskret)")
        plt.ylabel("Anteil falsch klassifiziert")
        plt.ylim([0, 1])
        # plt.title("Anteil Pfadsegmente falsch klassifiziert")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)
