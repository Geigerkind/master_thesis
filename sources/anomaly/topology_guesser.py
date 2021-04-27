class AnomalyTopologyGuesser:
    def __init__(self, topology):
        # Parameters
        self.topology = topology

        # Configuration
        # Controls how long it will output anomaly detected if it found one, no matter what
        self.alarm_duration = 5

        # Internal variables
        self.__duration_waited = self.alarm_duration

    def predict(self, previous_distinct_location, current_location):
        if current_location > 0 and previous_distinct_location > 0 and self.topology[current_location][0] != previous_distinct_location:
            self.__duration_waited = 0

        # Emit the alarm if we are within an alarm duration
        if self.alarm_duration > self.__duration_waited:
            self.__duration_waited = self.__duration_waited + 1
            return True
        return False
