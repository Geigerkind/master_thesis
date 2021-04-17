import os

import numpy as np


class LogMetrics():
    def __init__(self, path, prefix, prediction):
        self.prefix = prefix
        self.prediction = prediction

        # Configuration
        self.file_path = path + prefix + "/"
        try:
            os.mkdir(self.file_path)
        except:
            pass

        # Write to the log file
        log_file = open(self.file_path + "log_general_metrics.csv", "w")
        log_file.write("location_change_frequency,location_change_frequency_not_zero,"
                       "average_confidence,average_confidence_not_zero,fraction_zero\n")
        log_file.write("{0},{1},{2},{3},{4}\n".format(
            self.__location_change_frequency(self.prediction),
            self.__location_change_frequency_not_zero(self.prediction),
            self.__average_confidence(self.prediction),
            self.__average_confidence_not_zero(self.prediction),
            self.__fraction_zero(self.prediction),
        ))
        log_file.close()

    def __location_change_frequency(self, predicted):
        location_changes = 0
        current_location = self.__get_discrete_label(predicted[0])
        for pred_label in predicted:
            new_location = self.__get_discrete_label(pred_label)
            if new_location != current_location:
                location_changes = location_changes + 1
                current_location = new_location
        return location_changes / len(predicted)

    def __location_change_frequency_not_zero(self, predicted):
        location_changes = 0
        total = 0
        current_location = self.__get_discrete_label(predicted[0])
        for pred_label in predicted:
            new_location = self.__get_discrete_label(pred_label)
            if new_location > 0:
                total = total + 1
                if new_location != current_location:
                    location_changes = location_changes + 1
                    current_location = new_location

        if total == 0:
            return -1
        return location_changes / total

    def __average_confidence(self, predicted):
        confidence = 0
        for prediction in predicted:
            confidence = confidence + prediction[self.__get_discrete_label(prediction)]
        return confidence / len(predicted)

    def __average_confidence_not_zero(self, predicted):
        confidence = 0
        total = 0
        for prediction in predicted:
            pred_label = self.__get_discrete_label(prediction)
            if pred_label > 0:
                confidence = confidence + prediction[pred_label]
                total = total + 1
        if total == 0:
            return -1
        return confidence / total

    def __fraction_zero(self, predicted):
        amount = 0
        for prediction in predicted:
            pred_label = self.__get_discrete_label(prediction)
            if pred_label == 0:
                amount = amount + 1
        return amount / len(predicted)

    def __get_discrete_label(self, label):
        return np.asarray(label).argmax()
