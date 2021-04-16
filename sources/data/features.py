from enum import Enum


# Features that are processed in the DataCompiler.
# Despite its name, it adds several features of the topic,
# eg. the standard deviation, max, min, mean of this kind of data
class Features(Enum):
    PreviousLocation = 0
    # LastDistinctLocation = 1
    # StandardDeviation = 2
    # Maximum = 3
    # Minimum = 4
    # Mean = 5
    AccessPointDetection = 6
    Temperature = 7
    Heading = 8
    Acceleration = 9
    Light = 10
    Volume = 11
    Time = 12
    Angle = 13
