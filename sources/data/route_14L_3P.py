import pandas as pd
import math


def get_route_14L_3P_labeled(proximity, sampling_frequency_factor):
    """
    :param proximity: Euclidean distance where a point should be counted towards a label
    :param sampling_frequency_factor: Multiple of 0.05
    :return: Labeled Route
    """

    # Find first occurrences of each location
    data = pd.read_csv("/home/shino/Uni/master_thesis/bin/data/14L_3P.csv")
    initial_positions = dict()
    for row in data.iterrows():
        if not (row[1]["pos"] in initial_positions):
            initial_positions[row[1]["pos"]] = row[1]

    # Label all points within the proximity
    def get_labeled_point(row):
        pt = None
        for point in initial_positions.values():
            distance = math.sqrt((row["x_pos"] - point["x_pos"]) ** 2 + (row["y_pos"] - point["y_pos"]) ** 2)
            if distance <= proximity:
                pt = point
                break

        if pt is None:
            row["location"] = 0
        else:
            row["location"] = pt["pos"]
        return row

    return data.apply(lambda row: get_labeled_point(row), axis=1).query(
        "t_stamp % " + str(sampling_frequency_factor * 0.05) + " == 0 or " + str(sampling_frequency_factor) + " == 1")
