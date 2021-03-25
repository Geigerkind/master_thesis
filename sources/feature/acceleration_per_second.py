# We assume a constant acceleration between two time steps and attempt to approximate the velocity
# We have to take into account that we will always experience velocity BUT we dont know if we slow down.
# Could be some kind of start, stop motion, like on a conveyor belt.
# So we just want to have a feeling for the velocity.
# So maybe not velocity but acceleration per second
# Assuming following data structure as input:
# [[timestamp, acc_x, acc_y, acc_z]]
class FeatureAccelerationPerSecond:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        input_len = len(self.input)
        if input_len == 0:
            return [0, 0, 0]

        index = input_len - 1
        last_ts = self.input[index][0]
        acceleration = [0, 0, 0]
        while index > 0 and last_ts - self.input[index][0] < 1:
            acceleration[0] = acceleration[0] + self.input[index][1]
            acceleration[1] = acceleration[1] + self.input[index][2]
            acceleration[2] = acceleration[2] + self.input[index][3]
            index = index - 1

        if last_ts - self.input[index][0] < 1:
            acceleration[0] / (last_ts - self.input[index][0])
            acceleration[1] / (last_ts - self.input[index][0])
            acceleration[2] / (last_ts - self.input[index][0])

        return acceleration
