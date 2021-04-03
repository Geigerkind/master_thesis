import os

import matplotlib.pyplot as plt
import numpy as np


class GraphPathSegmentMisclassified:
    def __init__(self, prefix, model, is_dt, test_features, test_labels):
        self.prefix = prefix
        self.model = model
        self.is_dt = is_dt
        self.test_features = np.asarray(test_features)
        self.test_labels = np.asarray(test_labels)

        # Configuration
        self.file_path = "/home/shino/Uni/master_thesis/bin/" + prefix + "/"
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
        predicted = self.model.predict(self.test_features)

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

            if self.__get_discrete_label(predicted[i]) != current_location:
                misclassified_count = misclassified_count + 1

        values = []
        for i in range(len(result)):
            if total[i] == 0:
                values.append(0)
            else:
                values.append(result[i] / total[i])

        return values

    def __generate_graph(self):
        y = self.__calculate_path_segment_misclassified()
        plt.bar(range(len(y)), y)
        plt.xlabel("Pfadsegment (Diskret)")
        plt.ylabel("Anteil falsch klassifiziert")
        plt.ylim([0, 1])
        plt.title("Anteil Pfadsegmente falsch klassifiziert")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
