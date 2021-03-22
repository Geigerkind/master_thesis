# The idea here is to infer meta information about the time since a certain last event
# For example Time since the last wakeup. This could indicate certain factory steps
class FeatureTimeSinceLast:
    def __init__(self, input, find_events_function, value_extraction_function):
        self.input = input
        self.find_events_function = find_events_function
        self.value_extraction_function = value_extraction_function
        self.feature = self.__calculate()

    def __calculate(self):
        extracted_values = [self.value_extraction_function(x) for x in self.input]
        event_indices = self.find_events_function(extracted_values)

        if len(event_indices) == 0:
            return 0

        return self.input[len(self.input) - 1][0] - self.input[event_indices[len(event_indices) - 1]][0]