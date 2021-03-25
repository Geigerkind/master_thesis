# Assuming following data structure:
# [[timestamp, acc_x, acc_y, acc_z]]
class FeatureAccelerationMomentum:
    def __init__(self, input):
        self.input = input

        # Configuration
        self.decay_factor = 0.9

        self.feature = self.__calculate()

    def __calculate(self):
        input_len = len(self.input)
        if input_len == 0:
            return [0, 0, 0]

        acceleration = [0, 0, 0]
        for i in range(len(self.input)):
            acceleration[0] = acceleration[0] * self.decay_factor + self.input[i][1]
            acceleration[1] = acceleration[1] * self.decay_factor + self.input[i][2]
            acceleration[2] = acceleration[2] * self.decay_factor + self.input[i][3]

        return acceleration
