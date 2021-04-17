import os

import matplotlib.pyplot as plt
import numpy as np


class GraphTrueVsPredicted:
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
        return "{0}_true_vs_predicted.png".format(self.prefix)

    def __generate_graph(self):
        y_true_position = self.test_labels
        y_predicted_position = self.prediction
        if not self.is_dt:
            y_true_position = [np.asarray(x).argmax() for x in self.test_labels]
            y_predicted_position = [np.asarray(x).argmax() for x in self.prediction]

        accuracy = 0
        accuracy_given_location_is_cont_the_same_and_within_5_entries = 0
        accuracy_given_location_is_cont_the_same_and_within_10_entries = 0
        for i in range(len(y_true_position)):
            if y_true_position[i] == y_predicted_position[i]:
                accuracy = accuracy + 1
                accuracy_given_location_is_cont_the_same_and_within_5_entries = accuracy_given_location_is_cont_the_same_and_within_5_entries + 1
                accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries + 1
            else:
                is_within_5 = True
                has_gotten_right_location = False
                # Check 5 before
                for j in range(i, max(i - 5, 0) - 1, -1):
                    is_within_5 = is_within_5 and y_predicted_position[i] == y_predicted_position[j]
                    has_gotten_right_location = has_gotten_right_location or y_true_position[j] == y_predicted_position[
                        j]
                    if has_gotten_right_location:
                        break

                if has_gotten_right_location and is_within_5:
                    accuracy_given_location_is_cont_the_same_and_within_5_entries = accuracy_given_location_is_cont_the_same_and_within_5_entries + 1
                    accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries + 1
                    continue

                is_within_10 = True
                has_gotten_right_location = False
                # Check 10 before
                for j in range(i, max(i - 10, 0) - 1, -1):
                    is_within_10 = is_within_10 and y_predicted_position[i] == y_predicted_position[j]
                    has_gotten_right_location = has_gotten_right_location or y_true_position[j] == y_predicted_position[
                        j]
                    if has_gotten_right_location:
                        break

                if has_gotten_right_location and is_within_10:
                    accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries + 1
                    continue

                is_within_5 = True
                has_gotten_right_location = False
                # Check 5 after
                for j in range(i, min(i + 5, len(y_true_position))):
                    is_within_5 = is_within_5 and y_predicted_position[i] == y_predicted_position[j]
                    has_gotten_right_location = has_gotten_right_location or y_true_position[j] == y_predicted_position[
                        j]
                    if has_gotten_right_location:
                        break

                if has_gotten_right_location and is_within_5:
                    accuracy_given_location_is_cont_the_same_and_within_5_entries = accuracy_given_location_is_cont_the_same_and_within_5_entries + 1
                    accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries + 1
                    continue

                is_within_10 = True
                has_gotten_right_location = False
                # Check 10 after
                for j in range(i, min(i + 10, len(y_true_position))):
                    is_within_10 = is_within_10 and y_predicted_position[i] == y_predicted_position[j]
                    has_gotten_right_location = has_gotten_right_location or y_true_position[j] == y_predicted_position[
                        j]
                    if has_gotten_right_location:
                        break

                if has_gotten_right_location and is_within_10:
                    accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries + 1
                    continue

        accuracy = accuracy / len(y_true_position)
        accuracy_given_location_is_cont_the_same_and_within_5_entries = accuracy_given_location_is_cont_the_same_and_within_5_entries / len(
            y_true_position)
        accuracy_given_location_is_cont_the_same_and_within_10_entries = accuracy_given_location_is_cont_the_same_and_within_10_entries / len(
            y_true_position)

        accuracy_given_previous_location_was_correct = 0
        total_prev_location_correct = 0
        for i in range(1, len(y_true_position)):
            if y_true_position[i - 1] == y_predicted_position[i - 1]:
                total_prev_location_correct = total_prev_location_correct + 1
                if y_true_position[i] == y_predicted_position[i]:
                    accuracy_given_previous_location_was_correct = accuracy_given_previous_location_was_correct + 1
        if total_prev_location_correct == 0:
            accuracy_given_previous_location_was_correct = -1
        accuracy_given_previous_location_was_correct = accuracy_given_previous_location_was_correct / total_prev_location_correct

        log_file = open(self.file_path + "log_true_vs_predicted.csv", "w")
        log_file.write("accuracy,accuracy_given_previous_location_was_correct,"
                       "accuracy_given_location_is_cont_the_same_and_within_5_entries,"
                       "accuracy_given_location_is_cont_the_same_and_within_10_entries\n")
        log_file.write("{0},{1},{2},{3}".format(accuracy, accuracy_given_previous_location_was_correct,
                                                accuracy_given_location_is_cont_the_same_and_within_5_entries,
                                                accuracy_given_location_is_cont_the_same_and_within_10_entries))
        log_file.close()

        fig, ax1 = plt.subplots()
        ax1.plot(range(len(self.prediction)), y_true_position, "o-g")
        ax1.plot(range(len(self.prediction)), y_predicted_position, "*-b")
        ax1.set_xlabel("Pfadeintrag (Diskret)")
        ax1.set_ylabel("Position (Diskret)")
        ax1.set_ylim([0, self.num_outputs - 1])
        ax1.set_title("Ist vs. Soll Position")
        fig.legend(['Ist Position', 'Soll Position'], loc=[0.68, 0.77])
        plt.savefig("{0}{1}".format(self.file_path, self.__graph_name()))
        plt.clf()
        plt.close(fig)
