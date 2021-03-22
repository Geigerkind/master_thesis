# The idea of this feature is to find patterns in the specified time frame
# For example hammering will have volume spikes in a periodic pattern
# The input may differ here, the subject_function will take care of for example extracting the maximum over time
# subject_function: [number] -> number
# Examples: max, min, avg
class FeatureAverageTimeBetween:
    def __init__(self, input, subject_function, delta, value_extraction_function):
        self.input = input
        self.subject_function = subject_function
        self.delta = delta
        self.value_extraction_function = value_extraction_function
        self.feature = self.__calculate()

    def __calculate(self):
        value_extracted_input = [self.value_extraction_function(x) for x in self.input]
        global_subject = self.subject_function(value_extracted_input)
        found_subjects = []
        for i in range(len(value_extracted_input)):
            if abs(global_subject - value_extracted_input[i]) < self.delta:
                found_subjects.append(self.input[i][0])

        if len(found_subjects) == 1:
            return found_subjects[0] - self.input[0][0]

        if len(found_subjects) == 0:
            return 0

        acc_time = 0
        for i in range(len(found_subjects) - 1):
            acc_time = acc_time + found_subjects[i + 1] - found_subjects[i]

        return acc_time / (len(found_subjects) - 1)
