class FeatureMean:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        return sum(self.input) / len(self.input)
