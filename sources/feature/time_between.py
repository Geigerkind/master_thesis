class FeatureTimeBetween:
    def __init__(self, input, subject_function, delta):
        self.input = input
        self.subject_function = subject_function
        self.delta = delta
        self.feature = self.__calculate()

    def __calculate(self):
        raise Exception("Not implemented.")
