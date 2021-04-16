import os

import numpy as np

from sources.ffnn.gen_ffnn import GenerateFFNN


class LogMetrics():
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

        # Predict
        predicted = 0
        if self.use_continued_prediction:
            predicted = self.model.continued_predict_proba(
                self.test_features) if self.is_dt else GenerateFFNN.static_continued_predict(self.model,
                                                                                             self.test_features,
                                                                                             self.num_outputs)
        else:
            predicted = self.model.predict_proba(self.test_features) if self.is_dt else self.model.predict(
                self.test_features)

        # Write to the log file
        log_file = open(self.file_path + "log_general_metrics.csv", "w")
        log_file.write("location_change_frequency,location_change_frequency_not_zero,"
                       "average_confidence,average_confidence_not_zero,fraction_zero\n")
        log_file.write("{0},{1},{2},{3},{4}\n".format(
            self.__location_change_frequency(predicted),
            self.__location_change_frequency_not_zero(predicted),
            self.__average_confidence(predicted),
            self.__average_confidence_not_zero(predicted),
            self.__fraction_zero(predicted),
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
