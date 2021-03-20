class FeatureAbsoluteMax:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        return max([abs(x) for x in self.input])
