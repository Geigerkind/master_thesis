import math

import pandas as pd

TEST_ROUTE_1_POINTS = [
    [-3.5, 0.7, "Top Left", 1],
    [3.5, 0.7, "Top Right", 2],
    [3.5, -0.7, "Bottom Right", 3],
    [-3.5, -0.7, "Bottom Left", 4],
    [-3.5, -0.2, "Detour Bottom Left", 5],
    [-3.5, 0.3, "Detour Top Left", 6],
    [-2.5, -0.2, "Detour Bottom Right", 7],
    [-2.5, 0.3, "Detour Top Right", 8],
]


def get_test_route_1_labeled_by_xy(is_pos_data, proximity):
    """
    :param is_pos_data: If true it uses pos_data.txt else run_data.txt
    :param proximity: Euclidean distance where a point should be counted towards a label:
    :return: Labeled Route
    """

    # Takes a row and returns row with label
    def get_labeled_point(row):
        label = 0
        for point in TEST_ROUTE_1_POINTS:
            distance = math.sqrt((row["x_pos"] - point[0]) ** 2 + (row["y_pos"] - point[1]) ** 2)
            if distance <= proximity:
                label = point[3]
                break
        row["label"] = label
        return row

    if is_pos_data:
        return pd.read_csv("/home/shino/Uni/master_thesis/external_sources/trial_route_1_data/pos_data.txt") \
            .apply(lambda row: get_labeled_point(row), axis=1)
    return pd.read_csv("/home/shino/Uni/master_thesis/external_sources/trial_route_1_data/run_data.txt") \
        .apply(lambda row: get_labeled_point(row), axis=1)
