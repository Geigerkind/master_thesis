import math

import pandas as pd

TEST_ROUTE_1_POINTS = [
    [-3.5, 0.7, "Top Left", 1, 3],
    [3.5, 0.7, "Top Right", 2, 1],
    [3.5, -0.7, "Bottom Right", 3, 4],
    [-3.5, -0.7, "Bottom Left", 4, 5],
    [-3.5, -0.2, "Detour Bottom Left", 5, 7],
    [-3.5, 0.3, "Detour Top Left", 6, 1],
    [-2.5, -0.2, "Detour Bottom Right", 7, 8],
    [-2.5, 0.3, "Detour Top Right", 8, 6],
]


def get_test_route_1_labeled_by_xy(is_pos_data, proximity, sampling_frequency_factor):
    """
    :param is_pos_data: If true it uses pos_data.txt else run_data.txt
    :param proximity: Euclidean distance where a point should be counted towards a label
    :param sampling_frequency_factor: Multiple of 0.05
    :return: Labeled Route
    """

    # Takes a row and returns row with label
    def get_labeled_point(row):
        pt = 0
        for point in TEST_ROUTE_1_POINTS:
            distance = math.sqrt((row["x_pos"] - point[0]) ** 2 + (row["y_pos"] - point[1]) ** 2)
            if distance <= proximity:
                pt = point
                break

        if pt == 0:
            row["location"] = 0
            row["prev_location"] = 0
        else:
            row["location"] = pt[3]
            row["prev_location"] = pt[4]
        return row

    path = "/home/shino/Uni/master_thesis/external_sources/trial_route_1_data/run_data.txt"
    if is_pos_data:
        path = "/home/shino/Uni/master_thesis/external_sources/trial_route_1_data/pos_data.txt"

    return pd.read_csv(path).apply(lambda row: get_labeled_point(row), axis=1).query(
        "t_stamp % " + str(sampling_frequency_factor * 0.05) + " == 0")
