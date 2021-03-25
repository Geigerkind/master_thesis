import numpy


# Assumes input [number] and returns the index of the absolute max
class FeatureDiscreteAbsMax:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        return numpy.array([abs(x) for x in self.input]).argmax()
