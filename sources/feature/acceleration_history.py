class FeatureAccelerationHistory:
    def __init__(self, input):
        self.input = input

        # Configuration:
        self.num_windows = 5
        self.window_rest_pattern = [
            [0, 0, 0, 0, 0],  # 0
            [0, 0, 1, 0, 0],  # 1
            [0, 1, 0, 1, 0],  # 2
            [1, 0, 1, 0, 1],  # 3
            [1, 1, 1, 0, 1],  # 4
        ]

        self.feature = self.__calculate()

    def __calculate(self):
        input_len = len(self.input)
        if input_len == 0:
            return [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        amount_per_window = input_len / self.num_windows
        rest = input_len % self.num_windows
        windows = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        current_window = 0
        for i in range(input_len):
            if i > (current_window + 1) * amount_per_window + self.window_rest_pattern[rest][current_window]:
                current_window = current_window + 1

            windows[current_window][0] = windows[current_window][0] + self.input[i][1][0]
            windows[current_window][1] = windows[current_window][1] + self.input[i][1][1]
            windows[current_window][2] = windows[current_window][2] + self.input[i][1][2]

        return windows
