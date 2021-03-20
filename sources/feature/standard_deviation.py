import numpy as np


class FeatureStandardDeviation:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        return np.nanstd(self.input)
