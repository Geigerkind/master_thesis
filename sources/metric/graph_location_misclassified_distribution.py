import os

import matplotlib.pyplot as plt
import numpy as np

from sources.ffnn.gen_ffnn import GenerateFFNN


class GraphLocationMisclassifiedDistribution:
    def __init__(self, path, prefix, model, is_dt, test_features, test_labels, num_outputs, use_continued_prediction):
        self.prefix = prefix
        self.model = model
        self.is_dt = is_dt
        self.test_features = np.asarray(test_features)
        self.test_labels = np.asarray(test_labels)
        self.num_outputs = num_outputs
        self.use_continued_prediction = use_continued_prediction

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_location_misclassified_distribution.png".format(self.prefix)

    def __get_discrete_label(self, label):
        if self.is_dt:
            return label
        return np.asarray(label).argmax()

    def __compare_predicted(self, test, predicted):
        if self.is_dt:
            return test == predicted
        return self.__get_discrete_label(test) == self.__get_discrete_label(predicted)

    def __calculate_location_misclassified(self):
        predicted = 0
        if self.use_continued_prediction:
            predicted = self.model.continued_predict(
                self.test_features) if self.is_dt else GenerateFFNN.static_continued_predict(self.model,
                                                                                             self.test_features,
                                                                                             self.num_outputs)
        else:
            predicted = self.model.predict(self.test_features)

        result = dict()
        for i in range(int(self.num_outputs)):
            result[i] = 0

        total = 0
        for i in range(len(self.test_labels)):
            real_label = self.__get_discrete_label(self.test_labels[i])
            if not self.__compare_predicted(self.test_labels[i], predicted[i]):
                result[real_label] = result[real_label] + 1
                total = total + 1

        keys = []
        values = []
        for key in result:
            keys.append(key)
            if total == 0:
                values.append(0)
            else:
                values.append(result[key] / total)
        return keys, values

    def __generate_graph(self):
        x, y = self.__calculate_location_misclassified()
        plt.bar(x, y)
        plt.xlabel("Ort (Diskret)")
        plt.ylabel("Anteil falsch klassifiziert")
        plt.ylim([0, 1])
        plt.title("Verteilung von Orte falsch klassifiziert")
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
