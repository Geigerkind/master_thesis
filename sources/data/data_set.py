from enum import Enum


# Data sets found in /bin/data
class DataSet(Enum):
    SimpleSquare = "simple_square.csv", "simple_square"
    ManyCorners = "many_corners.csv", "many_corners"
    LongRectangle = "long_rectangle.csv", "long_rectangle"
    RectangleWithRamp = "rectangle_with_ramp.csv", "rectangle_with_ramp"
    Anomaly = "14L_3P.csv", "14L_3P"
    AnomalySimpleSquareTrain1 = "simple_square_anomaly_train1.csv", "simple_square_anomaly_train1", [
        [[1.7, 2.7], [3.5, 4.1]]], "simple_square"
    AnomalySimpleSquareTrain2 = "simple_square_anomaly_train2.csv", "simple_square_anomaly_train2", [
        [[-1.3, 1.2], [0.1, 3.1]]], "simple_square"
    AnomalySimpleSquareTrain3 = "simple_square_anomaly_train3.csv", "simple_square_anomaly_train3", [
        [[4.7, 4], [6.5, 6.2]], [[3.0, 4.7], [4.7, 6.2]]], "simple_square"
    AnomalyLongRectangleTrain1 = "long_rectangle_anomaly_train1.csv", "long_rectangle_anomaly_train1", [
        [[4.8, 3], [7, 7]]], "long_rectangle"
    AnomalyLongRectangleTrain2 = "long_rectangle_anomaly_train2.csv", "long_rectangle_anomaly_train2", [
        [[-2, -1.5], [0.5, 3.9]]], "long_rectangle"
    AnomalyRectangleWithRampTrain1 = "rectangle_with_ramp_anomaly_train1.csv", "rectangle_with_ramp_anomaly_train1", [
        [[-1, -0.9], [1, 1]]], "rectangle_with_ramp"
    AnomalyManyCornersTrain1 = "many_corners_anomaly_train1.csv", "many_corners_anomaly_train1", [
        [[3.8, 0], [6.5, 0.5]]], "many_corners"
