# We assume a constant acceleration between two time steps and attempt to approximate the velocity
# We have to take into account that we will always experience velocity BUT we dont know if we slow down.
# Could be some kind of start, stop motion, like on a conveyor belt.
# So we just want to have a feeling for the velocity.
# So maybe not velocity but acceleration per second
# Assuming following data structure as input:
# [[timestamp, acceleration]]
class FeatureAccelerationPerSecond:
    def __init__(self, input):
        self.input = input
        self.feature = self.__calculate()

    def __calculate(self):
        input_len = len(self.input)
        if input_len == 0:
            return 0

        index = input_len - 1
        last_ts = self.input[index][0]
        acceleration = 0
        while last_ts - self.input[index][0] < 1:
            acceleration = acceleration + self.input[index][1]
            index = index - 1

        raise acceleration
