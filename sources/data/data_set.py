from enum import Enum


# Data sets found in /bin/data
class DataSet(Enum):
    SimpleSquare = "simple_square.csv", "simple_square"
    ManyCorners = "many_corners.csv", "many_corners"
    LongRectangle = "long_rectangle.csv", "long_rectangle"
    RectangleWithRamp = "rectangle_with_ramp.csv", "rectangle_with_ramp"
    Anomaly = "14L_3P.csv", "14L_3P"
    AnomalyTrain1 = "simple_square_anomaly_train1.csv", "simple_square_anomaly_train1"
    AnomalyTrain2 = "simple_square_anomaly_train2.csv", "simple_square_anomaly_train2"
    AnomalyTrain3 = "simple_square_anomaly_train3.csv", "simple_square_anomaly_train3"
