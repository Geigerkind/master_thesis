class FeatureAbsoluteMin:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        return min([abs(x) for x in self.input])
