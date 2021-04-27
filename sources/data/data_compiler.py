import copy
import math
import os
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from sources.config import BIN_FOLDER_PATH, NUM_CORES
from sources.data.data_set import DataSet
from sources.data.features import Features
from sources.feature.max import FeatureMax
from sources.feature.mean import FeatureMean
from sources.feature.min import FeatureMin
from sources.feature.standard_deviation import FeatureStandardDeviation

# Random seed initialization
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)


####################################
# Methods used for parallelization #
####################################
# Those have been added here,
# because they cant be pickled
# as class object

def par_lrd_adjust_pos(input_args):
    def adjust_pos(ad_row, offset):
        ad_row["pos"] = ad_row["pos"] + 1
        ad_row["original_pos"] = ad_row["pos"]
        ad_row["pos"] = ad_row["pos"] + offset
        return ad_row

    df, offset = input_args
    return df.apply(lambda i_row: adjust_pos(i_row, offset), axis=1)


def par_lrd_set_location(input_args):
    def set_location(row, args):
        initial_positions, proximity = args
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

    df, args = input_args
    return df.apply(lambda i_row: set_location(i_row, args), axis=1)


def par_ef_calculate_features(args):
    data, i, window_size, features, lookback_window = args
    result = []

    for lw_i in range(lookback_window):
        window_offset = int(window_size * (lw_i + 1))
        window = data.iloc[(i - window_offset):(i - window_offset + window_size), :]

        # x_acc_col_list = window["x_acc"].tolist()
        # y_acc_col_list = window["y_acc"].tolist()
        # z_acc_col_list = window["z_acc"].tolist()
        acc_total_abs_col_list = (window["x_acc"] + window["y_acc"] + window["z_acc"]).abs().tolist()
        ang_total_abs_col_list = (window["x_ang"] + window["y_ang"] + window["z_ang"]).abs().tolist()
        # x_ang_col_list = window["x_ang"].tolist()
        # y_ang_col_list = window["y_ang"].tolist()
        # z_ang_col_list = window["z_ang"].tolist()
        light_col_list = window["light"].tolist()
        temperature_col_list = window["temperature"].tolist()
        heading_col_list = window["heading"].tolist()
        volume_col_list = window["volume"].tolist()
        time_col_list = window["t_stamp"].tolist()

        prev_location = 0
        current_location = window.iloc[window_size - 1]["location"]
        for j in range(i - window_offset + window_size, 0):
            if current_location != data.iloc[j]["location"] and data.iloc[j]["location"] > 0:
                prev_location = data.iloc[j]["location"]
                break

        # NOTE: Changes to the features also require changes in __populate_feature_name_map()
        if Features.PreviousLocation in features:
            result.append(window.iloc[window_size - 2]["location"])
            result.append(prev_location)

        if Features.Acceleration in features:
            result.append(acc_total_abs_col_list[-1])
            result.append(FeatureStandardDeviation(acc_total_abs_col_list).feature)
            result.append(FeatureMax(acc_total_abs_col_list).feature)
            result.append(FeatureMin(acc_total_abs_col_list).feature)
            result.append(FeatureMean(acc_total_abs_col_list).feature)

        if Features.Light in features:
            result.append(light_col_list[-1])
            result.append(FeatureStandardDeviation(light_col_list).feature)
            result.append(FeatureMax(light_col_list).feature)
            result.append(FeatureMin(light_col_list).feature)
            result.append(FeatureMean(light_col_list).feature)

        if Features.AccessPointDetection in features:
            result.append(int(window.iloc[window_size - 1]["access_point_0"]))
            result.append(int(window.iloc[window_size - 1]["access_point_1"]))
            result.append(int(window.iloc[window_size - 1]["access_point_2"]))
            result.append(int(window.iloc[window_size - 1]["access_point_3"]))
            result.append(int(window.iloc[window_size - 1]["access_point_4"]))

        if Features.Temperature in features:
            result.append(temperature_col_list[-1])
            result.append(FeatureStandardDeviation(temperature_col_list).feature)
            result.append(FeatureMax(temperature_col_list).feature)
            result.append(FeatureMin(temperature_col_list).feature)
            result.append(FeatureMean(temperature_col_list).feature)

        if Features.Heading in features:
            result.append(heading_col_list[-1])
            result.append(FeatureStandardDeviation(heading_col_list).feature)
            result.append(FeatureMax(heading_col_list).feature)
            result.append(FeatureMin(heading_col_list).feature)
            result.append(FeatureMean(heading_col_list).feature)

        if Features.Volume in features:
            result.append(volume_col_list[-1])
            result.append(FeatureStandardDeviation(volume_col_list).feature)
            result.append(FeatureMax(volume_col_list).feature)
            result.append(FeatureMin(volume_col_list).feature)
            result.append(FeatureMean(volume_col_list).feature)

        if Features.Time in features:
            result.append(FeatureStandardDeviation(time_col_list).feature)
            # Time since last interrupt
            # result.append(time_col_list[window_size - 1] - time_col_list[window_size - 2])
            # Time since last discrete position changed
            # TODO: This requires changes to the prediction data relabeling algo
            """
            time_since = 0
            for j in range(i, 0):
                if prev_location == data.iloc[j]["location"]:
                    time_since = time_col_list[window_size - 1] - data.iloc[j]["t_stamp"]
                    break
            result.append(time_since)
            """

        if Features.Angle in features:
            result.append(ang_total_abs_col_list[-1])
            result.append(FeatureStandardDeviation(ang_total_abs_col_list).feature)
            result.append(FeatureMax(ang_total_abs_col_list).feature)
            result.append(FeatureMin(ang_total_abs_col_list).feature)
            result.append(FeatureMean(ang_total_abs_col_list).feature)

    # TODO: Ist das mitm lbwindow > 1 noch korrekt?
    window = data.iloc[(i - window_size):i, :]
    return window.iloc[window_size - 1]["cycle"], window.iloc[window_size - 1]["location"], result


def par_process_data_set(args):
    data_set, count, total_len, is_verbose, sampling_interval, training_sampling_rate_in_location, max_training_cycle, in_live_mode = args
    if is_verbose:
        print("Processing data set {0} of {1}".format(count, total_len))

    profiling_interrupt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def induces_interrupt(store, cmp_row, keys, threshold, cmp_function, prof_key):
        if cmp_function(store, cmp_row, keys, threshold):
            profiling_interrupt[prof_key] = profiling_interrupt[prof_key] + 1
            for key in keys:
                store[key] = cmp_row[key]
            return True
        return False

    def induces_interrupt_single(store, cmp_row, key, threshold, cmp_function, prof_key):
        if cmp_function(store, cmp_row, key, threshold):
            profiling_interrupt[prof_key] = profiling_interrupt[prof_key] + 1
            store[key] = cmp_row[key]
            return True
        return False

    def f_cmp_abs_diff(r1, r2, keys, threshold):
        r1_sum = 0
        r2_sum = 0
        for key in keys:
            r1_sum = r1_sum + r1[key]
            r2_sum = r2_sum + r2[key]
        r1_sum = abs(r1_sum)
        r2_sum = abs(r2_sum)
        return abs(r1_sum - r2_sum) >= threshold * r2_sum

    def f_cmp_abs_diff_abs_threshold(r1, r2, key, threshold):
        if abs(r1[key] - r2[key]) >= threshold:
            return True
        return False

    def f_cmp_abs_diff_threshold_of_last(r1, r2, key, threshold):
        if abs(r1[key] - r2[key]) >= threshold * abs(r1[key]):
            return True
        return False

    def f_cmp_any_unequal(r1, r2, keys, _threshold):
        for key in keys:
            if r1[key] != r2[key]:
                return True
        return False

    # last interrupt
    li = data_set.iloc[0].copy()

    # new_df = DataFrame(columns=data_set.keys())
    index_map = []
    index = 0

    # Optimization cause python is slow af
    key_set1 = ["x_acc", "y_acc", "z_acc"]
    key_set2 = ["x_ang", "y_ang", "z_ang"]
    key3 = "heading"
    key4 = "temperature"
    key5 = "light"
    key6 = "volume"
    key_set7 = ["access_point_0", "access_point_1", "access_point_2", "access_point_3", "access_point_4"]
    key8 = "x_ang"
    key9 = "y_ang"
    key10 = "z_ang"

    rows = []
    for row in data_set.iterrows():
        if induces_interrupt(li, row[1], key_set1, 0.1, f_cmp_abs_diff, 0) \
                or induces_interrupt_single(li, row[1], key8, 0.01, f_cmp_abs_diff_abs_threshold, 7) \
                or induces_interrupt_single(li, row[1], key9, 0.01, f_cmp_abs_diff_abs_threshold, 8) \
                or induces_interrupt_single(li, row[1], key10, 0.1, f_cmp_abs_diff_abs_threshold, 9) \
                or induces_interrupt_single(li, row[1], key3, 8, f_cmp_abs_diff_abs_threshold, 2) \
                or induces_interrupt_single(li, row[1], key4, 0.12, f_cmp_abs_diff_threshold_of_last, 3) \
                or induces_interrupt_single(li, row[1], key5, 0.12, f_cmp_abs_diff_threshold_of_last, 4) \
                or induces_interrupt_single(li, row[1], key6, 0.16, f_cmp_abs_diff_threshold_of_last, 5) \
                or induces_interrupt(li, row[1], key_set7, 0, f_cmp_any_unequal, 6):
            index_map.append(index)
            rows.append(row[1])
        """
        # Hat sich gezeigt, dass zusammen mit einem Datenfenster dies die Klassifizierungsgenauigkeit in der Praxis verschlechtert.
        # Besser ist eine bessere Kalibrierung des Interrupt-Systems.
        elif (row[1]["location"] > 0 and row[1]["cycle"] <= max_training_cycle and not in_live_mode and
              row[1]["t_stamp"] % training_sampling_rate_in_location == 0):
            found_one = False
            for i in range(len(rows) - 1, 0, -1):
                if row[1]["t_stamp"] - rows[i]["t_stamp"] <= training_sampling_rate_in_location:
                    if rows[i]["location"] == row[1]["location"]:
                        found_one = True
                        break
                else:
                    break
            if found_one:
                index_map.append(index)
                rows.append(row[1])
        """

        index = index + 1

    print("{0} => {1}".format(count, profiling_interrupt))
    if is_verbose:
        print("Reduced the data set " + str(count) + " by: %.2f Percent" % (100 * (1 - (len(rows) / len(data_set)))))
        print("Finished processing data set {0} of {1}".format(count, total_len))
    return DataFrame(rows, columns=data_set.keys()), index_map


class DataCompiler:
    def __init__(self, data_sets, features, train_with_faulty_data=False, encode_paths_between_as_location=False,
                 use_synthetic_routes=False, proximity=0.1, manual_data_set=None, manual_num_inputs=0,
                 manual_num_outputs=0):
        """
        This tool compiles provided data sets with given features into training data for decision trees and knn.
        It extracts features, encodes the locations, creates faulty sets and creates synthetic routes.
        It attempts to utilize all processing power available to cope for the massive influx of data as
        fast as possible.

        :param data_sets: Array of data sets that should be processed, see "DataSet"-Enum
        :param features: Set of features that should be used, see "Features"-Enum
        :param train_with_faulty_data: If "True", adds faulty data to the training sets
        :param encode_paths_between_as_location: Encodes the paths between locations as locations
        :param use_synthetic_routes: Adds synthetic routes to the training data
        :param proximity: The proximity in which a location around the first labeled location is considered this location
        """
        # Input variables
        self.data_sets = data_sets
        self.use_synthetic_routes = use_synthetic_routes
        self.features = features
        self.train_with_faulty_data = train_with_faulty_data
        self.encode_paths_between_as_location = encode_paths_between_as_location
        self.proximity = proximity
        self.manual_data_set = manual_data_set

        # Configuration
        self.num_cycles = 20
        self.num_validation_cycles = 5
        self.num_warmup_cycles = 5
        self.window_size = 3
        self.lookback_window = 1  # NOT FULLY IMPLEMENTED!
        # Not used atm.
        self.sampling_interval = 1  # Every second there is at least on sampling
        # Ensure that we have enough training samples for the training data
        self.sampling_interval_in_location_for_training_data = 0.25
        self.fraction_fault_training_data = 0.1

        # Internal configuration
        self.__num_temporary_test_sets = 6  # Note the anomaly set added at the load
        self.__using_manual_data_set = not (manual_data_set is None)
        self.__is_verbose = not self.__using_manual_data_set

        # Declarations
        self.__reference_locations = dict()
        self.__data_sets = dict()
        self.__raw_data = []
        self.index_maps = []

        self.location_neighbor_graph = dict()
        self.position_map = dict()
        self.access_point_range = 1.5
        self.access_point_positions = [
            [0, 0],
            [4, 4],
            [4, 2],
            [3.5, 0.5],
            [1, 2.5]
        ]
        self.ambient_temperature = 20  # Degrees Celsius
        # ([x,y], temperature, distance_until_ambient_is_reached_again)
        self.heat_sources = [
            ([0, 1], 26, 2),
            ([3, 2], 10, 2),
            ([1, 3], 100, 2),
            ([4, 4.5], 40, 1)
        ]
        self.magnetic_sources = [
            ([3, 4], 2),
            ([0, 4], 1.5),
            ([3, 1.5], 3)
        ]
        self.background_noise_mean = 20
        self.background_noise_variance = 3
        # ([x, y], max_volume, distance_until_ambient, is_constant, [periodicity?])
        self.noises = [
            ([2, 3.5], 60, 2, False, 1.25),
            ([1, 0], 100, 2, False, 1.5),
            ([0, 1.5], 40, 2, True)
        ]

        self.num_outputs = 0
        self.num_inputs = 0
        self.name_map_features = []
        self.name_map_data_sets_result = []
        self.name_map_data_sets_test = []
        self.name_map_data_sets_faulty = []
        self.name_map_data_sets_faulty_test = []
        self.name_map_data_sets_temporary = []
        self.__populate_features_name_map()

        self.test_raw_data = []
        self.test_features_dt = []
        self.test_features_knn = []
        self.test_labels_dt = []
        self.test_labels_knn = []

        self.result_raw_data = []
        self.result_features_dt = []
        self.result_features_knn = []
        self.result_labels_dt = []
        self.result_labels_knn = []

        self.faulty_raw_data = []
        self.faulty_features_dt = []
        self.faulty_features_knn = []
        self.faulty_labels_dt = []
        self.faulty_labels_knn = []

        self.faulty_test_raw_data = []
        self.faulty_test_features_dt = []
        self.faulty_test_features_knn = []
        self.faulty_test_labels_dt = []
        self.faulty_test_labels_knn = []

        self.temporary_test_set_raw_data = []
        self.temporary_test_set_features_dt = []
        self.temporary_test_set_features_knn = []
        self.temporary_test_set_labels_dt = []
        self.temporary_test_set_labels_knn = []

        if not (DataSet.SimpleSquare in self.data_sets) and not self.__using_manual_data_set:
            raise Exception("At least Simple Square must be in the data sets!")

        # Execute the compiler steps
        if self.__using_manual_data_set:
            self.__raw_data = [self.manual_data_set]
            self.num_inputs = manual_num_inputs
            self.num_outputs = manual_num_outputs
        else:
            self.num_outputs = self.__load_raw_data() + 1
            self.__raw_data = [x for x in self.__data_sets.values()] + self.__generate_synthetic_routes()

        if not self.__using_manual_data_set:
            self.__create_temporary_test_sets()
        self.__add_synthetic_sensor_data()
        self.__interrupt_based_selection_cmp_prev_interrupt()
        if not self.__using_manual_data_set:
            self.__remove_temporary_test_sets()
            self.__create_faulty_data_sets()
        self.__extract_features()
        self.result_raw_data = self.__raw_data
        if not self.__using_manual_data_set:
            self.__create_faulty_route_with_skipped_locations()
        self.__configure_variables()

    def __populate_features_name_map(self):
        if Features.PreviousLocation in self.features:
            self.name_map_features.append("previous_location")
            self.name_map_features.append("previous_distinct_location")

        if Features.Acceleration in self.features:
            self.name_map_features.append("acc_last")  #
            self.name_map_features.append("acc_std")
            self.name_map_features.append("acc_max")
            self.name_map_features.append("acc_min")  #
            self.name_map_features.append("acc_mean")

        if Features.Light in self.features:
            self.name_map_features.append("light_last")  #
            self.name_map_features.append("light_std")
            self.name_map_features.append("light_max")
            self.name_map_features.append("light_min")  #
            self.name_map_features.append("light_mean")

        if Features.AccessPointDetection in self.features:
            self.name_map_features.append("ap_0")
            self.name_map_features.append("ap_1")
            self.name_map_features.append("ap_2")
            self.name_map_features.append("ap_3")
            self.name_map_features.append("ap_4")

        if Features.Temperature in self.features:
            self.name_map_features.append("temperature_last")  #
            self.name_map_features.append("temperature_std")
            self.name_map_features.append("temperature_max")
            self.name_map_features.append("temperature_min")
            self.name_map_features.append("temperature_mean")

        if Features.Heading in self.features:
            self.name_map_features.append("heading_last")  #
            self.name_map_features.append("heading_std")
            self.name_map_features.append("heading_max")  #
            self.name_map_features.append("heading_min")  #
            self.name_map_features.append("heading_mean")  #

        if Features.Volume in self.features:
            self.name_map_features.append("volume_last")  #
            self.name_map_features.append("volume_std")
            self.name_map_features.append("volume_max")  #
            self.name_map_features.append("volume_min")  #
            self.name_map_features.append("volume_mean")  #

        if Features.Time in self.features:
            self.name_map_features.append("time_std")

        if Features.Angle in self.features:
            self.name_map_features.append("ang_last")  #
            self.name_map_features.append("ang_std")
            self.name_map_features.append("ang_max")
            self.name_map_features.append("ang_min")  #
            self.name_map_features.append("ang_mean")  #

    def __configure_variables(self):
        self.__raw_data = 0
        self.__data_sets = 0

        if not self.__using_manual_data_set:
            self.num_inputs = len(self.result_features_knn[0][0][0])

        if self.train_with_faulty_data:
            # Use X% of each location for training data
            f_dt = []
            f_knn = []
            l_dt = []
            l_knn = []
            for fault_data_set_index in range(len(self.faulty_labels_dt)):
                i_f_dt = []
                i_f_knn = []
                i_l_dt = []
                i_l_knn = []

                # Create index maps for each location
                for cycle in range(self.num_cycles):
                    i_f_dt.append([])
                    i_f_knn.append([])
                    i_l_dt.append([])
                    i_l_knn.append([])
                    index_map = dict()
                    fault_data_set = self.faulty_labels_dt[fault_data_set_index][cycle]
                    for label_index in range(len(fault_data_set)):
                        label = fault_data_set[label_index]
                        if label in index_map:
                            index_map[label].append(label_index)
                        else:
                            index_map[label] = []

                    for i_map in index_map.values():
                        # For each location draw a permutation
                        permutation = np.random.permutation(len(i_map))

                        # For 10% of the permutation vector add them to the result dataset
                        for i in range(int(math.ceil(len(i_map) / 10))):
                            i_f_dt[cycle].append(self.faulty_features_dt[fault_data_set_index][cycle][permutation[i]])
                            i_f_knn[cycle].append(self.faulty_features_knn[fault_data_set_index][cycle][permutation[i]])
                            i_l_dt[cycle].append(self.faulty_labels_dt[fault_data_set_index][cycle][permutation[i]])
                            i_l_knn[cycle].append(self.faulty_labels_knn[fault_data_set_index][cycle][permutation[i]])

                # Finally add it to the result set
                f_dt.append(i_f_dt)
                f_knn.append(i_f_knn)
                l_dt.append(i_l_dt)
                l_knn.append(i_l_knn)

            self.result_features_dt = self.result_features_dt + f_dt
            self.result_features_knn = self.result_features_knn + f_knn
            self.result_labels_dt = self.result_labels_dt + l_dt
            self.result_labels_knn = self.result_labels_knn + l_knn

    def __remove_temporary_test_sets(self):
        for _ in range(self.__num_temporary_test_sets):
            self.temporary_test_set_raw_data.append(self.__raw_data.pop())

    def __create_temporary_test_sets(self):
        # Reverse ordering because its removed in reverse order
        self.name_map_data_sets_temporary.append("combined_test_route")
        self.name_map_data_sets_temporary.append("anomaly2")
        self.name_map_data_sets_temporary.append("anomaly_train3")
        self.name_map_data_sets_temporary.append("anomaly_train2")
        self.name_map_data_sets_temporary.append("anomaly_train1")
        self.name_map_data_sets_temporary.append("anomaly1")

        set1 = self.__glue_routes_together(DataSet.SimpleSquare, DataSet.Anomaly, 5)
        set2 = self.__create_combined_test_route()

        # temporarily add it to raw data such that sensor data is added
        self.__raw_data.append(set1)
        self.__raw_data.append(set2)

    def __generate_synthetic_routes(self):
        # Generate synthetic routes by gluing routes together and adjusting timestamps accordingly.
        synthetic_routes = []
        # NOTE: xyz-pos has not been adjusted!
        if self.use_synthetic_routes:
            if DataSet.SimpleSquare in self.data_sets:
                if DataSet.ManyCorners in self.data_sets:
                    synthetic_routes.append(self.__glue_routes_together(DataSet.SimpleSquare, DataSet.ManyCorners, 3))

                if DataSet.LongRectangle in self.data_sets:
                    synthetic_routes.append(self.__glue_routes_together(DataSet.SimpleSquare, DataSet.LongRectangle, 5))

                if DataSet.RectangleWithRamp in self.data_sets:
                    synthetic_routes.append(
                        self.__glue_routes_together(DataSet.SimpleSquare, DataSet.RectangleWithRamp, 2))
        return synthetic_routes

    def __load_raw_data(self):
        def parallelize(data, func, args):
            cores = NUM_CORES
            map_args = []
            data_split = np.array_split(data, cores)
            for split in data_split:
                map_args.append([split, args])
            pool = Pool(cores)
            data = pd.concat(pool.map(func, map_args))
            pool.close()
            pool.join()
            return data

        anomaly_data_sets = [DataSet.Anomaly, DataSet.AnomalyTrain1, DataSet.AnomalyTrain2, DataSet.AnomalyTrain3]

        location_offset = 0
        for data_set in self.data_sets + anomaly_data_sets:
            if self.__is_verbose:
                print("Loading Dataset: {0}".format(data_set.value[0]))
            self.__data_sets[data_set] = pd.read_csv(BIN_FOLDER_PATH + "/data/" + data_set.value[0])

            if data_set in anomaly_data_sets and data_set != DataSet.Anomaly:
                def is_row_in_anomaly(row):
                    for area in data_set.value[2]:
                        if area[0][0] <= row["x_pos"] <= area[1][0] and area[0][1] <= row["y_pos"] <= area[1][1]:
                            return True
                    return False

                self.__data_sets[data_set]["is_anomaly"] = self.__data_sets[data_set].apply(is_row_in_anomaly, axis=1)
            else:
                self.__data_sets[data_set]["is_anomaly"] = data_set == DataSet.Anomaly

            start_location_offset = location_offset

            test_set = 0
            if not (data_set in anomaly_data_sets):
                test_set = pd.read_csv(BIN_FOLDER_PATH + "/data/" + data_set.value[1] + "_test.csv")

            if not (data_set in anomaly_data_sets):
                self.name_map_data_sets_result.append(data_set.value[1])
                self.name_map_data_sets_test.append(data_set.value[1] + "_test")

            if self.__is_verbose:
                print("Adjusting pos...")
            adjust_pos_offset = location_offset if len(data_set.value) == 2 else self.__reference_locations[data_set.value[3]][1]
            self.__data_sets[data_set] = parallelize(self.__data_sets[data_set], par_lrd_adjust_pos,
                                                     adjust_pos_offset)
            if not (data_set in anomaly_data_sets):
                test_set = parallelize(test_set, par_lrd_adjust_pos, location_offset)

            if len(data_set.value) == 2:
                location_offset = self.__data_sets[data_set]["pos"].max()

            if self.__is_verbose:
                print("Setting Location...")

            def append_to_neighbor_graph(dict, left, right):
                if not (left in dict):
                    dict[left] = []
                if not (right in dict[left]):
                    dict[left].append(right)

            initial_positions = dict()
            previous_pos = self.__data_sets[data_set].iloc[0]["pos"]
            for row in self.__data_sets[data_set].iterrows():
                # There is no 0 pos to worry about here
                if row[1]["pos"] != previous_pos and not (data_set in anomaly_data_sets):
                    append_to_neighbor_graph(self.location_neighbor_graph, row[1]["pos"], previous_pos)
                    # append_to_neighbor_graph(self.location_neighbor_graph, previous_pos, row[1]["pos"])
                    previous_pos = row[1]["pos"]

                if not (row[1]["pos"] in initial_positions):
                    initial_positions[row[1]["pos"]] = row[1]
                    if not (data_set in anomaly_data_sets):
                        self.position_map[row[1]["pos"]] = [row[1]["x_pos"], row[1]["y_pos"]]
            self.__data_sets[data_set] = parallelize(self.__data_sets[data_set], par_lrd_set_location,
                                                     [initial_positions, self.proximity])
            if not (data_set in anomaly_data_sets):
                test_set = parallelize(test_set, par_lrd_set_location, [initial_positions, self.proximity])

            if self.encode_paths_between_as_location:
                if self.__is_verbose:
                    print("Label paths between locations...")
                # IMPORTANT: This location mapping assumes Circles!
                # The anomaly data set is ignored because the locations are not used anyway during evaluation!
                location_map = dict()
                previous_non_zero_pos = 0
                init_offset = location_offset if len(data_set.value) == 2 else self.__reference_locations[data_set.value[3]][1]
                for row in self.__data_sets[data_set].iterrows():
                    if not row[1]["is_anomaly"]:
                        if row[1]["location"] == 0:
                            row[1]["location"] = location_map[previous_non_zero_pos]
                        else:
                            if not (row[1]["location"] in location_map):
                                init_offset = init_offset + 1
                                location_map[row[1]["location"]] = init_offset
                                if not (data_set in anomaly_data_sets):
                                    # TODO: This does not make sense!
                                    # self.position_map[location_offset] = [row[1]["x_pos"], row[1]["y_pos"]]
                                    append_to_neighbor_graph(self.location_neighbor_graph, row[1]["location"],
                                                             location_offset)
                                    # append_to_neighbor_graph(self.location_neighbor_graph, location_offset,
                                    #                          row[1]["location"])
                            previous_non_zero_pos = row[1]["location"]

                if len(data_set.value) == 2:
                    location_offset = init_offset

                if not (data_set in anomaly_data_sets):
                    previous_non_zero_pos = 0
                    for row in test_set.iterrows():
                        if row[1]["location"] == 0:
                            row[1]["location"] = location_map[previous_non_zero_pos]
                        else:
                            row[1]["location"] = location_map[row[1]["location"]]
                            previous_non_zero_pos = row[1]["location"]

            if not (data_set in anomaly_data_sets):
                self.test_raw_data.append(test_set)

            if len(data_set.value) == 2:
                self.__reference_locations[data_set.value[1]] = [start_location_offset, location_offset]
        return location_offset

    def __create_combined_test_route(self):
        glued = self.__data_sets[DataSet.SimpleSquare].copy(deep=True)
        if DataSet.ManyCorners in self.data_sets:
            glued = self.__glue_routes_together(DataSet.SimpleSquare, DataSet.ManyCorners, 3, glued)
        if DataSet.LongRectangle in self.data_sets:
            glued = self.__glue_routes_together(DataSet.SimpleSquare, DataSet.LongRectangle, 5, glued)
        if DataSet.ManyCorners in self.data_sets:
            glued = self.__glue_routes_together(DataSet.SimpleSquare, DataSet.RectangleWithRamp, 2, glued)
        return glued

    def __create_faulty_data_sets(self):
        if self.__is_verbose:
            print("Creating faulty data sets...")

        def process_faulty_sets(data_set, count, ref_raw_data, ref_name_map, ref_orig_name_map, num_cycles, is_verbose):
            # Permute paths randomly
            if is_verbose:
                print("Creating permuted paths set...")
            result_permutation = DataFrame()
            for cycle in range(num_cycles):
                cycle_view = data_set.query("cycle == " + str(cycle))
                permutation = np.random.permutation(10)
                split_dataset = np.array_split(cycle_view, 10)
                for index in permutation:
                    result_permutation = result_permutation.append(split_dataset[index], ignore_index=False)

            ref_raw_data.append(result_permutation)
            ref_name_map.append("faulty_" + ref_orig_name_map[count] + "_permuted_paths")

            # Set sensor values 0
            if is_verbose:
                print("Creating nulled acceleration set...")
            nulled_acceleration = data_set.copy(deep=True)
            nulled_acceleration["x_acc"] = 0
            nulled_acceleration["y_acc"] = 0
            nulled_acceleration["z_acc"] = 0
            ref_raw_data.append(nulled_acceleration)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_nulled_acceleration")

            if is_verbose:
                print("Creating nulled angle set...")
            nulled_angle = data_set.copy(deep=True)
            nulled_angle["x_ang"] = 0
            nulled_angle["y_ang"] = 0
            nulled_angle["z_ang"] = 0
            ref_raw_data.append(nulled_angle)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_nulled_angle")

            if is_verbose:
                print("Creating nulled light set...")
            nulled_light = data_set.copy(deep=True)
            nulled_light["light"] = 0
            ref_raw_data.append(nulled_light)
            ref_name_map.append("faulty_" + ref_orig_name_map[count] + "_nulled_light")

            if is_verbose:
                print("Creating nulled access point set...")
            nulled_access_point = data_set.copy(deep=True)
            nulled_access_point["access_point_0"] = False
            nulled_access_point["access_point_1"] = False
            nulled_access_point["access_point_2"] = False
            nulled_access_point["access_point_3"] = False
            nulled_access_point["access_point_4"] = False
            ref_raw_data.append(nulled_access_point)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_nulled_access_point")

            if is_verbose:
                print("Creating nulled heading set...")
            nulled_heading = data_set.copy(deep=True)
            nulled_heading["heading"] = 0
            ref_raw_data.append(nulled_heading)
            ref_name_map.append("faulty_" + ref_orig_name_map[count] + "_nulled_heading")

            if is_verbose:
                print("Creating nulled temperature set...")
            nulled_temperature = data_set.copy(deep=True)
            nulled_temperature["temperature"] = 0
            ref_raw_data.append(nulled_temperature)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_nulled_temperature")

            if is_verbose:
                print("Creating nulled volume set...")
            nulled_volume = data_set.copy(deep=True)
            nulled_volume["volume"] = 0
            ref_raw_data.append(nulled_volume)
            ref_name_map.append("faulty_" + ref_orig_name_map[count] + "_nulled_volume")

            if is_verbose:
                print("Creating random acceleration deviations set...")
            max_deviation = 10  # in percent
            acc_deviation = data_set.copy(deep=True)
            acc_deviation["x_acc"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            acc_deviation["y_acc"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            acc_deviation["z_acc"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(acc_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_acceleration_deviation")

            max_deviation = 5  # in percent
            if is_verbose:
                print("Creating random angle deviations set...")
            ang_deviation = data_set.copy(deep=True)
            ang_deviation["x_ang"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ang_deviation["y_ang"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ang_deviation["z_ang"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(ang_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_angle_deviation")

            if is_verbose:
                print("Creating access point randomly not detected set...")
            chance_to_detect = 80  # in percent
            access_point_random_not_detect = data_set.copy(deep=True)
            access_point_random_not_detect["access_point_0"].apply(
                lambda x: x and random.randint(0, 100) <= chance_to_detect)
            access_point_random_not_detect["access_point_1"].apply(
                lambda x: x and random.randint(0, 100) <= chance_to_detect)
            access_point_random_not_detect["access_point_2"].apply(
                lambda x: x and random.randint(0, 100) <= chance_to_detect)
            access_point_random_not_detect["access_point_3"].apply(
                lambda x: x and random.randint(0, 100) <= chance_to_detect)
            access_point_random_not_detect["access_point_4"].apply(
                lambda x: x and random.randint(0, 100) <= chance_to_detect)
            ref_raw_data.append(access_point_random_not_detect)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_access_point_not_detect")

            if is_verbose:
                print("Creating random heading deviations set...")
            heading_deviation = data_set.copy(deep=True)
            heading_deviation["heading"].apply(
                lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(heading_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_heading_deviation")

            if is_verbose:
                print("Creating random light deviations set...")
            light_deviation = data_set.copy(deep=True)
            light_deviation["light"].apply(lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(light_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_light_deviation")

            if is_verbose:
                print("Creating random temperature deviations set...")
            temperature_deviation = data_set.copy(deep=True)
            temperature_deviation["temperature"].apply(
                lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(temperature_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_temperature_deviation")

            if is_verbose:
                print("Creating random volume deviations set...")
            volume_deviation = data_set.copy(deep=True)
            volume_deviation["volume"].apply(
                lambda x: x * ((100 + random.randint(-max_deviation, max_deviation)) / 100))
            ref_raw_data.append(volume_deviation)
            ref_name_map.append(
                "faulty_" + ref_orig_name_map[count] + "_random_volume_deviation")

        count = 0
        for data_set in self.__raw_data:
            process_faulty_sets(data_set, count, self.faulty_raw_data, self.name_map_data_sets_faulty,
                                self.name_map_data_sets_result, self.num_cycles, self.__is_verbose)
            count = count + 1

        count = 0
        for data_set in self.test_raw_data:
            process_faulty_sets(data_set, count, self.faulty_test_raw_data, self.name_map_data_sets_faulty_test,
                                self.name_map_data_sets_test, self.num_cycles, self.__is_verbose)
            count = count + 1

    def __create_faulty_route_with_skipped_locations(self):
        if Features.PreviousLocation in self.features:
            for data_set_index in range(len(self.result_labels_dt)):
                # Build a map of last distinct locations that are not 0
                last_distinct_locations_dt = []
                last_distinct_locations_knn = []

                f_copy_dt = copy.copy(self.result_features_dt[data_set_index])
                f_copy_knn = copy.copy(self.result_features_knn[data_set_index])

                for cycle in range(self.num_cycles):
                    for label_index in range(len(self.result_labels_dt[data_set_index][cycle])):
                        label = self.result_labels_dt[data_set_index][cycle][label_index]
                        if label > 0 and (
                                len(last_distinct_locations_dt) == 0 or last_distinct_locations_dt[-1] != label):
                            last_distinct_locations_dt.append(label)
                            last_distinct_locations_knn.append(label / (self.num_outputs - 1))

                        if label == 0 and len(last_distinct_locations_dt) >= 2:
                            f_copy_dt[cycle][label_index][1] = last_distinct_locations_dt[-2]
                            f_copy_knn[cycle][label_index][1] = last_distinct_locations_knn[-2]
                        elif label > 0 and len(last_distinct_locations_dt) >= 3:
                            f_copy_dt[cycle][label_index][1] = last_distinct_locations_dt[-3]
                            f_copy_knn[cycle][label_index][1] = last_distinct_locations_knn[-3]

                self.faulty_raw_data.append(copy.copy(self.result_raw_data[data_set_index]))
                self.faulty_labels_dt.append(copy.copy(self.result_labels_dt[data_set_index]))
                self.faulty_labels_knn.append(copy.copy(self.result_labels_knn[data_set_index]))
                self.faulty_features_dt.append(f_copy_dt)
                self.faulty_features_knn.append(f_copy_knn)
                self.name_map_data_sets_faulty.append(
                    "faulty_" + self.name_map_data_sets_result[data_set_index] + "_skipped_location")

            # Copy pasta cause I was lazy
            for data_set_index in range(len(self.test_labels_dt)):
                # Build a map of last distinct locations that are not 0
                last_distinct_locations_dt = []
                last_distinct_locations_knn = []

                f_copy_dt = copy.copy(self.test_features_dt[data_set_index])
                f_copy_knn = copy.copy(self.test_features_knn[data_set_index])

                for cycle in range(self.num_cycles):
                    for label_index in range(len(self.test_labels_dt[data_set_index][cycle])):
                        label = self.test_labels_dt[data_set_index][cycle][label_index]
                        if label > 0 and (
                                len(last_distinct_locations_dt) == 0 or last_distinct_locations_dt[-1] != label):
                            last_distinct_locations_dt.append(label)
                            last_distinct_locations_knn.append(label / (self.num_outputs - 1))

                        if label == 0 and len(last_distinct_locations_dt) >= 2:
                            f_copy_dt[cycle][label_index][1] = last_distinct_locations_dt[-2]
                            f_copy_knn[cycle][label_index][1] = last_distinct_locations_knn[-2]
                        elif label > 0 and len(last_distinct_locations_dt) >= 3:
                            f_copy_dt[cycle][label_index][1] = last_distinct_locations_dt[-3]
                            f_copy_knn[cycle][label_index][1] = last_distinct_locations_knn[-3]

                self.faulty_test_raw_data.append(copy.copy(self.test_raw_data[data_set_index]))
                self.faulty_test_labels_dt.append(copy.copy(self.test_labels_dt[data_set_index]))
                self.faulty_test_labels_knn.append(copy.copy(self.test_labels_knn[data_set_index]))
                self.faulty_test_features_dt.append(f_copy_dt)
                self.faulty_test_features_knn.append(f_copy_knn)
                self.name_map_data_sets_faulty_test.append(
                    "faulty_" + self.name_map_data_sets_test[data_set_index] + "_skipped_location")

    def __add_synthetic_sensor_data(self):
        if self.__is_verbose:
            print("Adding synthetic sensor data...")
        #############################
        # Detect WLAN Access Points #
        #############################
        # Model:
        # Each access point has a range
        # If the robot is within range, it detects the access point
        # For simplicity, we only consider x and y and using euclidean distance
        # All sampled routes are within [-3,7]^2
        if self.__is_verbose:
            print("Adding access point detection data...")

        for data_set in self.__raw_data:
            for ap in range(len(self.access_point_positions)):
                data_set["access_point_{0}".format(ap)] = ((data_set["x_pos"] - self.access_point_positions[ap][
                    0]) ** 2 + (
                                                                   data_set["y_pos"] - self.access_point_positions[ap][
                                                               1]) ** 2).apply(
                    lambda x: math.sqrt(x)) <= self.access_point_range

        if not self.__using_manual_data_set:
            for data_set in self.test_raw_data:
                for ap in range(len(self.access_point_positions)):
                    data_set["access_point_{0}".format(ap)] = ((data_set["x_pos"] - self.access_point_positions[ap][
                        0]) ** 2 + (
                                                                       data_set["y_pos"] -
                                                                       self.access_point_positions[ap][1]) ** 2).apply(
                        lambda x: math.sqrt(x)) <= self.access_point_range

        ###############
        # Temperature #
        ###############
        # Model:
        # We have an ambient temperature and heat sources.
        # Heat sources can be above or below ambient temperature.
        # The temperature is approaching the ambient temperature quadratically after a predefined distance.
        # We take then the temperature with the maximum absolute difference to the ambient, if there is a conflict.
        if self.__is_verbose:
            print("Adding temperature data...")

        def calculate_temperature(row):
            temps = []
            for heat_source in self.heat_sources:
                distance = ((row["x_pos"] - heat_source[0][0]) ** 2 + (
                        row["y_pos"] - heat_source[0][1]) ** 2)
                if distance > heat_source[2]:
                    temps.append(self.ambient_temperature)
                else:
                    amplitude = (heat_source[1] - self.ambient_temperature) / (heat_source[2] ** 2)
                    temps.append(self.ambient_temperature + amplitude * (distance ** 2))

            return temps[np.asarray([abs(x - self.ambient_temperature) for x in temps]).argmax()]

        for data_set in self.__raw_data:
            data_set["temperature"] = data_set.apply(calculate_temperature, axis=1)

        if not self.__using_manual_data_set:
            for data_set in self.test_raw_data:
                data_set["temperature"] = data_set.apply(calculate_temperature, axis=1)

        ############################
        # Magnetic field / compass #
        ############################
        # Model:
        # Simplification: We only consider xy, i.e. we get a heading in [0, 359]
        # north is 0 we always get the relative pos to north.
        # west is 270, east 90, south 180
        # if we move towards east we get 90
        # There are magnetic sources that have a certain strength to affect the natural heading.
        # Depending on the strength it will change the heading depending on the current position to the position
        # of the object emitting a force.
        # Eg. An influence of 50% will set the relative base position to wherever the strength is coming from
        # The object will have 3 parameters
        # ([x, y], total_range_of_influence)
        # While the percentage of influence reduces with the range quadratically, until it is after
        # total_range_of_influence not affecting the sensor anymore.
        # BIG NOTE:
        # The compass will face wherever the robot is pointing. Meaning that the heading is dependent on the
        # facing direction, which is generally bad. The feature must abstract over this, e.g. only consider the
        # absolute change or so
        # In a conveyor belt system the facing of the sensor will usually not change, because there are usually no round
        # conveyor belts. BUT it will not always be put on the conveyor belt with the same facing. Therefore, each
        # cycle, a random facing in [0, 359] will be drawn.
        if self.__is_verbose:
            print("Adding heading data...")

        facings = np.random.randint(0, 359, self.num_cycles)

        def calculate_heading(row):
            facing = facings[int(row["cycle"])]
            # Find an influence
            # NOTE: We assume no overlapping !!!!
            for source in self.magnetic_sources:
                distance = ((row["x_pos"] - source[0][0]) ** 2 + (
                        row["y_pos"] - source[0][1]) ** 2)
                # If we are within the influence of the magnetic source
                if distance <= source[1]:
                    amplitude = -1 / (source[1] ** 2)
                    influence = amplitude * (distance ** 2) + 1

                    # Translate source object into coordinate system with current position as center
                    tmo = [source[0][0] - row["x_pos"], source[0][1] - row["y_pos"]]

                    # Decide the quadrant: Clockwise: 0, 1, 2, 3
                    num_quadrant = 0
                    alpha = 0
                    if tmo[0] < 0:
                        if tmo[1] < 0:
                            alpha = math.asin(abs(tmo[1]) / math.sqrt((tmo[0] ** 2) + (tmo[1] ** 2)))
                            num_quadrant = 2
                        else:
                            alpha = math.asin(abs(tmo[0]) / math.sqrt((tmo[0] ** 2) + (tmo[1] ** 2)))
                            num_quadrant = 3
                    elif tmo[0] == 0:
                        if tmo[1] == 0:
                            # Its on top of the current pos
                            # No effect technically
                            return facing
                        elif tmo[1] < 0:
                            num_quadrant = 2
                        else:
                            # angle 0, quadrant 0
                            # Same cause it points where our magnetic north is defined
                            return facing
                    else:
                        if tmo[1] < 0:
                            alpha = math.asin(abs(tmo[0]) / math.sqrt((tmo[0] ** 2) + (tmo[1] ** 2)))
                            num_quadrant = 1
                        else:
                            alpha = math.asin(abs(tmo[1]) / math.sqrt((tmo[0] ** 2) + (tmo[1] ** 2)))

                    beta = num_quadrant * 90 + alpha
                    if num_quadrant >= 2:
                        return (facing + (360 * (1 + influence) - beta * (1 - influence))) % 360
                    return (facing + (360 - beta * (1 - influence))) % 360
            return facing

        for data_set in self.__raw_data:
            data_set["heading"] = data_set.apply(calculate_heading, axis=1)

        if not self.__using_manual_data_set:
            for data_set in self.test_raw_data:
                data_set["heading"] = data_set.apply(calculate_heading, axis=1)

        #################
        # Volume sensor #
        #################
        # Model:
        # Background Noise:
        # The volume tends to have a background noise that is constant to its mean.
        # Meaning we kind of have volume that is around a value with a certain variance.
        # This noise will be modeled with a uniform distribution.
        # Frequency Bands:
        # We can listen in several frequency bands.
        # Old people tend to not hear high frequencies, whereas younger people do.
        # We can use that to our advantage and output volume in different frequency bands.
        # For now not considered. TODO: This can be added later.
        # Volume over distance:
        # Things are initially louder and get quieter the further it travels until we eventually dont hear it any
        # more. The best example is hammering, where we can hear a loud spike and it deafens fast.
        # This will always be modeled. Constant noises will have a noise according to the distance,
        # periodic noises will have that as well with the initial spike but their deafening will also be modeled.
        # Periodic events vs. constant events:
        # Imagine hammering vs. the noise of an electric saw
        # Periodic events have a periodicity and occur according to that. The deafening spikes will be modeled.
        # Constant noise will be modeled as constant. Eventual interference is not considered.
        # Interference:
        # Noise tends to interfere with each other. They can destroy each other and resonate.
        # Unrealistic assumption: No interference, because we assume that they deafened so much after a while that
        # they dont interfere with each other
        if self.__is_verbose:
            print("Adding volume data...")

        def calculate_volume(row):
            # NOTE: We assume no overlapping !!!!
            background_noise = self.background_noise_mean + (
                    -self.background_noise_variance + (self.background_noise_variance * 2) * random.random())
            for noise in self.noises:
                distance = ((row["x_pos"] - noise[0][0]) ** 2 + (
                        row["y_pos"] - noise[0][1]) ** 2)

                if distance <= noise[2]:
                    amplitude = (background_noise - noise[1]) / (noise[2] ** 2)
                    if noise[3]:
                        # Constant noises
                        noise_volume = amplitude * (distance ** 2) + noise[1]
                        return noise_volume
                    else:
                        # Periodic noise
                        # We say that we can here its aftermath until after 0.5s
                        if row["t_stamp"] % noise[4] <= 0.5:
                            noise_volume = max((amplitude * (distance ** 2) + noise[1]) * (
                                    -4 * ((row["t_stamp"] % noise[4]) ** 2) + 1), background_noise)
                            return noise_volume
                        return background_noise
            return background_noise

        for data_set in self.__raw_data:
            data_set["volume"] = data_set.apply(calculate_volume, axis=1)

        if not self.__using_manual_data_set:
            for data_set in self.test_raw_data:
                data_set["volume"] = data_set.apply(calculate_volume, axis=1)

    def __interrupt_based_selection_cmp_prev_interrupt(self):
        # We collect data from all sensors if any of the sensors sends an interrupt
        # Therefore we define here for each row if it should fire an "interrupt"
        # compared to the previous row that fired an interrupt
        if self.__is_verbose:
            print("Filtering raw data by synthetic interrupts...")
        with Pool(processes=NUM_CORES) as pool:
            args = []
            count = 1
            for data_set in self.__raw_data:
                args.append([data_set, count, len(self.__raw_data), self.__is_verbose, self.sampling_interval,
                             self.sampling_interval_in_location_for_training_data,
                             self.num_cycles - self.num_validation_cycles - 1, self.__using_manual_data_set])
                count = count + 1
            self.__raw_data = []
            for res in pool.map(par_process_data_set, args):
                self.__raw_data.append(res[0])
                self.index_maps.append(res[1])

            args = []
            count = 1
            for data_set in self.test_raw_data:
                args.append([data_set, count, len(self.test_raw_data), self.__is_verbose, self.sampling_interval,
                             self.sampling_interval_in_location_for_training_data,
                             self.num_cycles - self.num_validation_cycles - 1, self.__using_manual_data_set])
                count = count + 1
            self.test_raw_data = []
            for res in pool.map(par_process_data_set, args):
                self.test_raw_data.append(res[0])
                self.index_maps.append(res[1])

    def __extract_features(self):
        # For each entry in the raw data array, extract features
        if self.__is_verbose:
            print("Extracting features...")
        sc = StandardScaler()

        def extract_from_data_sets(data_sets, window_size, input_features, input_num_outputs, lookback_window):
            count = 1
            result_features_dt = []
            result_features_knn = []
            result_labels_dt = []
            result_labels_knn = []
            for data_set in data_sets:
                if self.__is_verbose:
                    print("Processing data set {0} of {1}...".format(count, len(data_sets)))
                count = count + 1
                features_tmp = []
                labels_tmp = []
                cycles = []
                with Pool(processes=NUM_CORES) as pool:
                    args = []
                    for i in range((window_size * lookback_window) + 1, len(data_set)):
                        args.append([data_set, i, window_size, input_features, lookback_window])
                    result = pool.map(par_ef_calculate_features, args)
                    for (cycle, label, features) in result:
                        cycles.append(cycle)
                        labels_tmp.append(int(label))
                        features_tmp.append(features)

                if len(features_tmp) == 0:
                    result_features_dt.append([])
                    result_features_knn.append([])
                    result_labels_dt.append([])
                    result_labels_knn.append([])
                    continue

                if self.__is_verbose:
                    print("Normalizing KNN data...")
                knn_features_tmp = sc.fit_transform(features_tmp)

                if Features.PreviousLocation in input_features:
                    if self.__is_verbose:
                        print("Fixing location labels...")
                    for i in range(len(knn_features_tmp)):
                        # Manual scaling between 0 and 1
                        knn_features_tmp[i][0] = knn_features_tmp[i][0] / (input_num_outputs - 1)
                        knn_features_tmp[i][1] = knn_features_tmp[i][1] / (input_num_outputs - 1)

                if self.__is_verbose:
                    print("Onehot encoding KNN data...")

                def create_ohe_mapping(x, num_outputs):
                    mapping = []
                    for _ in range(x):
                        mapping.append(0)
                    mapping.append(1)
                    for _ in range(int(num_outputs) - 1 - x):
                        mapping.append(0)
                    return mapping

                knn_labels_tmp = [create_ohe_mapping(x, self.num_outputs) for x in labels_tmp]

                if self.__is_verbose:
                    print("Reshaping data...")
                dt_features = []
                dt_labels = []
                knn_features = []
                knn_labels = []

                current_cycle = 0
                cycle_dt_features = []
                cycle_dt_labels = []
                cycle_knn_features = []
                cycle_knn_labels = []
                for i in range(len(cycles)):
                    if cycles[i] > current_cycle:
                        current_cycle = cycles[i]
                        dt_features.append(cycle_dt_features)
                        dt_labels.append(cycle_dt_labels)
                        knn_features.append(cycle_knn_features)
                        knn_labels.append(cycle_knn_labels)
                        cycle_dt_features = []
                        cycle_dt_labels = []
                        cycle_knn_features = []
                        cycle_knn_labels = []

                    cycle_dt_features.append(features_tmp[i])
                    cycle_dt_labels.append(labels_tmp[i])
                    cycle_knn_features.append(knn_features_tmp[i])
                    cycle_knn_labels.append(knn_labels_tmp[i])

                dt_features.append(cycle_dt_features)
                dt_labels.append(cycle_dt_labels)
                knn_features.append(cycle_knn_features)
                knn_labels.append(cycle_knn_labels)

                result_features_dt.append(dt_features)
                result_features_knn.append(knn_features)
                result_labels_dt.append(dt_labels)
                result_labels_knn.append(knn_labels)
            return result_features_dt, result_features_knn, result_labels_dt, result_labels_knn

        if self.__is_verbose:
            print("Raw data...")
        result_features_dt, result_features_knn, result_labels_dt, result_labels_knn = extract_from_data_sets(
            self.__raw_data, self.window_size, self.features, self.num_outputs, self.lookback_window)
        self.result_labels_dt = result_labels_dt
        self.result_labels_knn = result_labels_knn
        self.result_features_dt = result_features_dt
        self.result_features_knn = result_features_knn

        if not self.__using_manual_data_set:
            if self.__is_verbose:
                print("Test data...")
            test_features_dt, test_features_knn, test_labels_dt, test_labels_knn = extract_from_data_sets(
                self.test_raw_data, self.window_size, self.features, self.num_outputs, self.lookback_window)
            self.test_labels_dt = test_labels_dt
            self.test_labels_knn = test_labels_knn
            self.test_features_dt = test_features_dt
            self.test_features_knn = test_features_knn

            if self.__is_verbose:
                print("Faulty data...")
            faulty_features_dt, faulty_features_knn, faulty_labels_dt, faulty_labels_knn = extract_from_data_sets(
                self.faulty_raw_data, self.window_size, self.features, self.num_outputs, self.lookback_window)
            self.faulty_labels_dt = faulty_labels_dt
            self.faulty_labels_knn = faulty_labels_knn
            self.faulty_features_dt = faulty_features_dt
            self.faulty_features_knn = faulty_features_knn

            if self.__is_verbose:
                print("Faulty test data...")
            faulty_test_features_dt, faulty_test_features_knn, faulty_test_labels_dt, faulty_test_labels_knn = extract_from_data_sets(
                self.faulty_test_raw_data, self.window_size, self.features, self.num_outputs, self.lookback_window)
            self.faulty_test_labels_dt = faulty_test_labels_dt
            self.faulty_test_labels_knn = faulty_test_labels_knn
            self.faulty_test_features_dt = faulty_test_features_dt
            self.faulty_test_features_knn = faulty_test_features_knn

            if self.__is_verbose:
                print("Temporary test sets data...")
            tr_features_dt, tr_features_knn, tr_labels_dt, tr_labels_knn = extract_from_data_sets(
                self.temporary_test_set_raw_data, self.window_size, self.features, self.num_outputs,
                self.lookback_window)
            self.temporary_test_set_labels_dt = tr_labels_dt
            self.temporary_test_set_labels_knn = tr_labels_knn
            self.temporary_test_set_features_dt = tr_features_dt
            self.temporary_test_set_features_knn = tr_features_knn

    def __glue_routes_together(self, data_set1, data_set2, glue_location, provided_route=None):
        pd.set_option('mode.chained_assignment', None)
        set1 = self.__data_sets[data_set1].copy(deep=True) if provided_route is None else provided_route.copy(deep=True)
        set2 = self.__data_sets[data_set2].copy(deep=True)

        route = DataFrame()
        set1_with_orig_glue_location = set1.query("cycle == 0").loc[set1.original_pos == glue_location].iloc[0][
            "location"]
        for cycle in range(self.num_cycles):
            set1_path_of_cycle = set1.query("cycle == " + str(cycle))
            set1_with_location_attached_to_orig_glue_location = set1_path_of_cycle.loc[
                set1.location == set1_with_orig_glue_location]

            set2_path_of_cycle = set2.query("cycle == " + str(cycle))
            set2_first_and_last_row = set2_path_of_cycle.iloc[[0, -1]]
            set2_time_between_start_and_end = set2_first_and_last_row.iloc[1]["t_stamp"] - \
                                              set2_first_and_last_row.iloc[0]["t_stamp"]

            set2_start_timestamp = set2_first_and_last_row.iloc[0]["t_stamp"]
            glue_location_start_ts = set1_with_location_attached_to_orig_glue_location.iloc[0]["t_stamp"]

            # Adjust timestamps of set2 path
            set2_path_of_cycle.loc[:, "t_stamp"] = set2_path_of_cycle.loc[:,
                                                   "t_stamp"] + glue_location_start_ts - set2_start_timestamp

            # Adjust timestamps of set1 remaining path
            second_glue_item_index = set1_with_location_attached_to_orig_glue_location.index[1]
            set1_path_of_cycle.loc[second_glue_item_index:,
            "t_stamp"] = set1_path_of_cycle.loc[second_glue_item_index:,
                         "t_stamp"] + set2_time_between_start_and_end

            # Adjust timestamps of set1 remaining cycles
            if cycle < self.num_cycles - 1:
                index = set1.query("cycle > " + str(cycle)).index[0]
                set1.loc[index:, "t_stamp"] = set1.loc[index:, "t_stamp"] + set2_time_between_start_and_end

            # Add routes in correct order into a new data_frame
            route = pd.concat(
                [route, set1_path_of_cycle.loc[:second_glue_item_index], set2_path_of_cycle,
                 set1_path_of_cycle.loc[second_glue_item_index:]], ignore_index=True)

        pd.set_option('mode.chained_assignment', 'warn')
        return route
