import os

import matplotlib.pyplot as plt
import numpy as np

from sources.ffnn.gen_ffnn import GenerateFFNN


class GraphTrueVsPredicted:
    def __init__(self, prefix, model, is_dt, test_features, test_labels, num_outputs, use_continued_prediction):
        self.prefix = prefix
        self.model = model
        self.is_dt = is_dt
        self.test_features = np.asarray(test_features)
        self.test_labels = np.asarray(test_labels)
        self.num_outputs = num_outputs
        self.use_continued_prediction = use_continued_prediction

        # Configuration
        self.file_path = "/home/shino/Uni/master_thesis/bin/" + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        self.__generate_graph()

    def __graph_name(self):
        return "{0}_true_vs_predicted.png".format(self.prefix)

    def __generate_graph(self):
        predicted = 0
        if self.use_continued_prediction:
            predicted = self.model.continued_predict(
                self.test_features) if self.is_dt else GenerateFFNN.static_continued_predict(self.model,
                                                                                             self.test_features,
                                                                                             self.num_outputs)
        else:
            predicted = self.model.predict(self.test_features)
        y_true_position = self.test_labels
        y_predicted_position = predicted
        if not self.is_dt:
            y_true_position = [np.asarray(x).argmax() for x in self.test_labels]
            y_predicted_position = [np.asarray(x).argmax() for x in predicted]

        fig, ax1 = plt.subplots()
        ax1.plot(range(len(predicted)), y_true_position, "o-g")
        ax1.plot(range(len(predicted)), y_predicted_position, "*-b")
        ax1.set_xlabel("Pfadeintrag (Diskret)")
        ax1.set_ylabel("Position (Diskret)")
        ax1.set_ylim([0, self.num_outputs - 1])
        ax1.set_title("Ist vs. Soll Position")
        fig.legend(['Ist Position', 'Soll Position'], loc=[0.68, 0.77])
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)
