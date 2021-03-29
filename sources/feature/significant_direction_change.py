# The idea of this feature is to find a pair (From, To) for a direction change.
# Ideally we want to know that we accelerated all the time towards east and then to west for example.
# We must be aware that there is a lot of noise, hence a delta parameter is used, to configure at which point
# we want to detect a "significant" change
# How it works:
# We can go 3^3 directions:
# Each XYZ can go in both ways and their combinations.
# We consider it a significant change, if from our rolling average
# a stronger change is detected than the input threshold
# Output: [None | True | False, None | True | False, None | True | False]
# Encoded as 0,1,2 respectively
# Assuming input: [[x_acc, y_acc, z_acc]]
class FeatureSignificantDirectionChange:
    def __init__(self, input, delta):
        self.input = input
        self.delta = delta

        # Configuration
        # Ignore gravitation
        self.acceleration_offset = [0, 0, 9.81]
        self.old_vec_fraction = 0.3  # Preserve a little from the old direction

        self.feature = self.__calculate()

    def __calculate(self):
        if len(self.input) == 0:
            return [0, 0, 0]

        # Initialization
        acc_vec = self.input[0]
        acc_vec[0] = acc_vec[0] + self.acceleration_offset[0]
        acc_vec[1] = acc_vec[1] + self.acceleration_offset[1]
        acc_vec[2] = acc_vec[2] + self.acceleration_offset[2]

        result = [0, 0, 0]

        for i in range(1, len(self.input)):
            for j in range(3):
                if abs(acc_vec[j] - self.input[i][j]) > self.delta:
                    result[j] = 1 if self.input[i][j] >= 0 else 2

                acc_vec[j] = self.old_vec_fraction * acc_vec[j] + (1 - self.old_vec_fraction) * self.input[i][j]

        return result
