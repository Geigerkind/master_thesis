from enum import Enum


# Data sets found in /bin/data
class DataSet(Enum):
    SimpleSquare = "simple_square.csv", "simple_square"
    ManyCorners = "many_corners.csv", "many_corners"
    LongRectangle = "long_rectangle.csv", "long_rectangle"
    RectangleWithRamp = "rectangle_with_ramp.csv", "rectangle_with_ramp"
    Anomaly = "14L_3P.csv", "14L_3P"
    AnomalyTrain1 = "simple_square_anomaly_train1.csv", "simple_square_anomaly_train1", [
        [[1.7, 2.7], [3.5, 4.1]]], "simple_square"
    AnomalyTrain2 = "simple_square_anomaly_train2.csv", "simple_square_anomaly_train2", [
        [[-1.3, 1.2], [0.1, 3.1]]], "simple_square"
    AnomalyTrain3 = "simple_square_anomaly_train3.csv", "simple_square_anomaly_train3", [[[4.7, 4], [6.5, 6.2]],
                                                                                         [[3.0, 4.7],
                                                                                          [4.7, 6.2]]], "simple_square"
