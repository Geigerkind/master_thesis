import math
import os
import random
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from sources.data.data_set import DataSet
from sources.data.features import Features
from sources.feature.max import FeatureMax
from sources.feature.mean import FeatureMean
from sources.feature.min import FeatureMin
from sources.feature.standard_deviation import FeatureStandardDeviation

np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)


def parallelize(data, func, args):
    cores = cpu_count()
    map_args = []
    data_split = np.array_split(data, cores)
    for split in data_split:
        map_args.append([split, args])
    pool = Pool(cores)
    data = pd.concat(pool.map(func, map_args))
    pool.close()
    pool.join()
    return data


def adjust_pos(ad_row, offset):
    ad_row["pos"] = ad_row["pos"] + 1
    ad_row["original_pos"] = ad_row["pos"]
    ad_row["pos"] = ad_row["pos"] + offset
    return ad_row


def apply_adjust_pos_to_df(input_args):
    df, offset = input_args
    return df.apply(lambda i_row: adjust_pos(i_row, offset), axis=1)


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


def apply_set_location_to_df(input_args):
    df, args = input_args
    return df.apply(lambda i_row: set_location(i_row, args), axis=1)


def calculate_features(args):
    data, i, window_size, features = args
    window = data.iloc[(i - window_size):i, :]

    # x_acc_col_list = window["x_acc"].tolist()
    # y_acc_col_list = window["y_acc"].tolist()
    # z_acc_col_list = window["z_acc"].tolist()
    acc_total_abs_col_list = (window["x_acc"] + window["y_acc"] + window["z_acc"]).abs().tolist()
    # x_ang_col_list = window["x_ang"].tolist()
    # y_ang_col_list = window["y_ang"].tolist()
    # z_ang_col_list = window["z_ang"].tolist()
    light_col_list = window["light"].tolist()
    temperature_col_list = window["temperature"].tolist()
    heading_col_list = window["heading"].tolist()
    volume_col_list = window["volume"].tolist()

    result = []
    if Features.PreviousLocation in features:
        result.append(window.iloc[window_size - 2]["location"])

        # if Features.LastDistinctLocation in features:
        prev_location = 0
        current_location = window.iloc[window_size - 1]["location"]
        for j in range(i, 0):
            if current_location != data.iloc[j]["location"] and data.iloc[j]["location"] > 0:
                prev_location = data.iloc[j]["location"]
                break

        result.append(prev_location)

    if Features.Acceleration in features:
        result.append(FeatureStandardDeviation(acc_total_abs_col_list).feature)
        result.append(FeatureMax(acc_total_abs_col_list).feature)
        result.append(FeatureMin(acc_total_abs_col_list).feature)
        result.append(FeatureMean(acc_total_abs_col_list).feature)

    if Features.Light in features:
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
        result.append(FeatureStandardDeviation(temperature_col_list).feature)
        result.append(FeatureMax(temperature_col_list).feature)
        result.append(FeatureMin(temperature_col_list).feature)
        result.append(FeatureMean(temperature_col_list).feature)

    if Features.Heading in features:
        result.append(FeatureStandardDeviation(heading_col_list).feature)
        result.append(FeatureMax(heading_col_list).feature)
        result.append(FeatureMin(heading_col_list).feature)
        result.append(FeatureMean(heading_col_list).feature)

    if Features.Volume in features:
        result.append(FeatureStandardDeviation(volume_col_list).feature)
        result.append(FeatureMax(volume_col_list).feature)
        result.append(FeatureMin(volume_col_list).feature)
        result.append(FeatureMean(volume_col_list).feature)

    return window.iloc[window_size - 1]["cycle"], window.iloc[window_size - 1]["location"], result


def process_data_set(args):
    data_set, count, total_len = args
    print("Processing data set {0} of {1}".format(count, total_len))
    # previous interrupt row
    pir = data_set.iloc[0]
    new_df = DataFrame(columns=data_set.keys())
    for row in data_set.iterrows():
        pir_total_acc = abs(pir["x_acc"] + pir["y_acc"] + pir["z_acc"])
        if (abs(abs(row[1]["x_acc"] + row[1]["y_acc"] + row[1]["z_acc"]) - pir_total_acc) >= 0.05 * pir_total_acc) \
                or (abs(pir["heading"] - row[1]["heading"]) >= 20) \
                or (abs(pir["temperature"] - row[1]["temperature"]) >= 0.15 * pir["temperature"]) \
                or (abs(pir["light"] - row[1]["light"]) >= 0.1 * pir["light"]) \
                or (pir["access_point_0"] != row[1]["access_point_0"]) \
                or (pir["access_point_1"] != row[1]["access_point_1"]) \
                or (pir["access_point_2"] != row[1]["access_point_2"]) \
                or (pir["access_point_3"] != row[1]["access_point_3"]) \
                or (pir["access_point_4"] != row[1]["access_point_4"]):
            pir = row[1]
            new_df = new_df.append(row[1], ignore_index=True)

    print("Reduced the data set " + str(count) + " by: %.2f Percent" % (100 * (1 - (len(new_df) / len(data_set)))))
    print("Finished processing data set {0} of {1}".format(count, total_len))
    return new_df


class DataCompiler:
    def __init__(self, data_sets, features, train_with_faulty_data=False, use_synthetic_routes=False, proximity=0.1):
        # Configuration
        self.num_cycles = 20
        self.num_validation_cycles = 5
        self.num_warmup_cycles = 3
        self.window_size = 5
        self.train_with_faulty_data = train_with_faulty_data

        # Declarations
        self.num_outputs = 0
        self.num_inputs = 0
        self.raw_data = []

        self.result_features_dt = []
        self.result_features_knn = []
        self.result_labels_dt = []
        self.result_labels_knn = []

        self.faulty_raw_data = []
        self.faulty_features_dt = []
        self.faulty_features_knn = []
        self.faulty_labels_dt = []
        self.faulty_labels_knn = []

        # Input variables
        self.data_sets = data_sets
        self.use_synthetic_routes = use_synthetic_routes
        self.features = features

        self.__data_sets = dict()
        location_offset = 0
        for data_set in self.data_sets:
            print("Loading Dataset: {0}".format(data_set.value))
            self.__data_sets[data_set] = pd.read_csv("/home/shino/Uni/master_thesis/bin/data/" + data_set.value)

            print("Adjusting pos...")
            self.__data_sets[data_set] = parallelize(self.__data_sets[data_set], apply_adjust_pos_to_df,
                                                     location_offset)
            location_offset = self.__data_sets[data_set]["pos"].max()

            print("Setting Location...")
            initial_positions = dict()
            for row in self.__data_sets[data_set].iterrows():
                if not (row[1]["pos"] in initial_positions):
                    initial_positions[row[1]["pos"]] = row[1]
            self.__data_sets[data_set] = parallelize(self.__data_sets[data_set], apply_set_location_to_df,
                                                     [initial_positions, proximity])

        # Generate synthetic routes by gluing routes together and adjusting timestamps accordingly.
        synthetic_routes = []
        # NOTE: xyz-pos has not been adjusted!
        if self.use_synthetic_routes:
            if DataSet.SimpleSquare in data_sets:
                if DataSet.ManyCorners in data_sets:
                    synthetic_routes.append(self.__glue_routes_together(DataSet.SimpleSquare, DataSet.ManyCorners, 3))

                if DataSet.LongRectangle in data_sets:
                    synthetic_routes.append(self.__glue_routes_together(DataSet.SimpleSquare, DataSet.LongRectangle, 5))

                if DataSet.RectangleWithRamp in data_sets:
                    synthetic_routes.append(
                        self.__glue_routes_together(DataSet.SimpleSquare, DataSet.RectangleWithRamp, 2))

        # Set raw data array
        raw_data = []
        for data_set in self.__data_sets.values():
            raw_data.append(data_set)
        raw_data = raw_data + synthetic_routes

        self.__data_sets = 0
        synthetic_routes = 0
        self.raw_data = raw_data
        raw_data = 0

        self.__add_synthetic_sensor_data()
        self.__interrupt_based_selection()
        self.__create_faulty_data_sets()

        self.num_outputs = location_offset + 1
        self.__extract_features()

        if self.train_with_faulty_data:
            self.result_features_dt = self.result_features_dt + self.faulty_features_dt
            self.result_features_knn = self.result_features_knn + self.faulty_features_knn
            self.result_labels_dt = self.result_labels_dt + self.faulty_labels_dt
            self.result_labels_knn = self.result_labels_knn + self.faulty_labels_knn

        self.num_inputs = len(self.result_features_knn[0][0][0])

    def __create_faulty_data_sets(self):
        for data_set in self.raw_data:

            # Permute paths randomly
            result_permutation = DataFrame()
            for cycle in range(self.num_cycles):
                cycle_view = data_set.query("cycle == " + str(cycle))
                permutation = np.random.permutation(10)
                split_dataset = np.array_split(cycle_view, 10)
                for index in permutation:
                    result_permutation = result_permutation.append(split_dataset[index], ignore_index=False)

            self.faulty_raw_data.append(result_permutation)

            # Set sensor values 0
            nulled_acceleration = data_set.copy(deep=True)
            nulled_acceleration["x_acc"] = 0
            nulled_acceleration["y_acc"] = 0
            nulled_acceleration["z_acc"] = 0
            self.faulty_raw_data.append(nulled_acceleration)

            nulled_light = data_set.copy(deep=True)
            nulled_light["light"] = 0
            self.faulty_raw_data.append(nulled_light)

            nulled_access_point = data_set.copy(deep=True)
            nulled_access_point["access_point_0"] = False
            nulled_access_point["access_point_1"] = False
            nulled_access_point["access_point_2"] = False
            nulled_access_point["access_point_3"] = False
            nulled_access_point["access_point_4"] = False
            self.faulty_raw_data.append(nulled_access_point)

            nulled_heading = data_set.copy(deep=True)
            nulled_heading["heading"] = 0
            self.faulty_raw_data.append(nulled_heading)

            nulled_temperature = data_set.copy(deep=True)
            nulled_temperature["temperature"] = 0
            self.faulty_raw_data.append(nulled_temperature)

            nulled_volume = data_set.copy(deep=True)
            nulled_volume["volume"] = 0
            self.faulty_raw_data.append(nulled_volume)

    def __add_synthetic_sensor_data(self):
        print("Adding synthetic sensor data...")
        #############################
        # Detect WLAN Access Points #
        #############################
        # Model:
        # Each access point has a range
        # If the robot is within range, it detects the access point
        # For simplicity, we only consider x and y and using euclidean distance
        # All sampled routes are within [-3,7]^2
        print("Adding access point detection data...")
        access_point_range = 1.5
        access_point_positions = [
            [-1, -1],
            [5, 5],
            [-1, 5],
            [5, -1],
            [3, 3]
        ]

        for data_set in self.raw_data:
            for ap in range(len(access_point_positions)):
                data_set["access_point_{0}".format(ap)] = ((data_set["x_pos"] - access_point_positions[ap][0]) ** 2 + (
                        data_set["y_pos"] - access_point_positions[ap][1]) ** 2).apply(
                    lambda x: math.sqrt(x)) <= access_point_range

        ###############
        # Temperature #
        ###############
        # Model:
        # We have an ambient temperature and heat sources.
        # Heat sources can be above or below ambient temperature.
        # The temperature is approaching the ambient temperature quadratically after a predefined distance.
        # We take then the temperature with the maximum absolute difference to the ambient, if there is a conflict.
        ambient_temperature = 20  # Degrees Celsius
        # ([x,y], temperature, distance_until_ambient_is_reached_again)
        heat_sources = [
            ([0, 0], 26, 2),
            ([3, 2], 10, 2),
            ([0, 4], 100, 2),
            ([5, 5], 40, 1)
        ]

        def calculate_temperature(row):
            temps = []
            for heat_source in heat_sources:
                distance = ((row["x_pos"] - heat_source[0][0]) ** 2 + (
                        row["y_pos"] - heat_source[0][1]) ** 2)
                if distance > heat_source[2]:
                    temps.append(ambient_temperature)
                else:
                    amplitude = (heat_source[1] - ambient_temperature) / (heat_source[2] ** 2)
                    temps.append(ambient_temperature + amplitude * (distance ** 2))

            return temps[np.asarray([abs(x - ambient_temperature) for x in temps]).argmax()]

        for data_set in self.raw_data:
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
        magnetic_sources = [
            ([3, 5], 2),
            ([0, 5], 1.5),
            ([3, 0], 3)
        ]

        facings = np.random.randint(0, 359, self.num_cycles)

        def calculate_heading(row):
            facing = facings[int(row["cycle"])]
            # Find an influence
            # NOTE: We assume no overlapping !!!!
            for source in magnetic_sources:
                distance = ((row["x_pos"] - source[0][0]) ** 2 + (
                        row["y_pos"] - source[0][1]) ** 2)
                # If we are within the influence of the magnetic source
                if distance <= source[1]:
                    amplitude = -1 / (source[1] ** 2)
                    influence = amplitude * (distance ** 2) + 1

                    # Translate source object into coordinate system with current position as center
                    trans_ms_object = [source[0][0] - row["x_pos"], source[0][1] - row["y_pos"]]

                    # Decide the quadrant: Clockwise: 0, 1, 2, 3
                    quadrant = 0
                    if trans_ms_object[0] < 0:
                        if trans_ms_object[1] < 0:
                            quadrant = 2
                        else:
                            quadrant = 3
                    elif trans_ms_object[0] == 0:
                        if trans_ms_object[1] == 0:
                            # Its on top of the current pos
                            # No effect technically
                            return facing
                        elif trans_ms_object[1] < 0:
                            # angle 0, quadrant 2
                            return (360 - (influence * 180) + facing) % 360
                        else:
                            # angle 0, quadrant 0
                            # Same cause it points where our magnetic north is defined
                            return facing
                    else:
                        if trans_ms_object[1] < 0:
                            quadrant = 1
                        else:
                            quadrant = 0

                    alpha = math.asin(
                        abs(trans_ms_object[0]) / math.sqrt((trans_ms_object[0] ** 2) + (trans_ms_object[1] ** 2)))
                    return (360 - (influence * ((quadrant * 90) + alpha)) + facing) % 360

            return facing

        for data_set in self.raw_data:
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
        background_noise_mean = 20
        background_noise_variance = 3
        # ([x, y], max_volume, distance_until_ambient, is_constant, [periodicity?])
        noises = [
            ([2, 5], 60, 2, False, 1.25),
            ([1, 0], 100, 2, False, 1.5),
            ([0, 5], 40, 2, True)
        ]

        def calculate_volume(row):
            # NOTE: We assume no overlapping !!!!
            background_noise = background_noise_mean + (
                    -background_noise_variance + (background_noise_variance * 2) * random.random())
            for noise in noises:
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
                            noise_volume = (amplitude * (distance ** 2) + noise[1]) * (
                                    -4 * ((row["t_stamp"] % noise[4]) ** 2) + 1)
                            return noise_volume
                        return background_noise
            return background_noise

        for data_set in self.raw_data:
            data_set["volume"] = data_set.apply(calculate_volume, axis=1)

    def __interrupt_based_selection(self):
        # We collect data from all sensors if any of the sensors sends an interrupt
        # Therefore we define here for each row if it should fire an "interrupt"
        # compared to the previous row that fired an interrupt
        print("Filtering raw data by synthetic interrupts...")
        with Pool(processes=cpu_count()) as pool:
            args = []
            count = 1
            for data_set in self.raw_data:
                args.append([data_set, count, len(self.raw_data)])
                count = count + 1
            self.raw_data = pool.map(process_data_set, args)

    def __extract_features(self):
        # For each entry in the raw data array, extract features
        print("Extracting features...")
        sc = StandardScaler()

        def extract_from_data_sets(data_sets, window_size, input_features, input_num_outputs):
            count = 1
            result_features_dt = []
            result_features_knn = []
            result_labels_dt = []
            result_labels_knn = []
            for data_set in data_sets:
                print("Processing data set {0} of {1}...".format(count, len(data_sets)))
                count = count + 1
                features_tmp = []
                labels_tmp = []
                cycles = []
                with Pool(processes=cpu_count()) as pool:
                    args = []
                    for i in range(window_size + 1, len(data_set)):
                        args.append([data_set, i, window_size, input_features])
                    result = pool.map(calculate_features, args)
                    for (cycle, label, features) in result:
                        cycles.append(cycle)
                        labels_tmp.append(int(label))
                        features_tmp.append(features)

                print("Normalizing KNN data...")
                knn_features_tmp = sc.fit_transform(features_tmp)

                if Features.PreviousLocation in input_features:
                    print("Fixing location labels...")
                    for i in range(len(knn_features_tmp)):
                        # Manual scaling between 0 and 1
                        knn_features_tmp[i][0] = knn_features_tmp[i][0] * (1 / input_num_outputs)
                        knn_features_tmp[i][1] = knn_features_tmp[i][1] * (1 / input_num_outputs)

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

        print("Raw data...")
        result_features_dt, result_features_knn, result_labels_dt, result_labels_knn = extract_from_data_sets(
            self.raw_data, self.window_size, self.features, self.num_outputs)
        self.result_labels_dt = result_labels_dt
        self.result_labels_knn = result_labels_knn
        self.result_features_dt = result_features_dt
        self.result_features_knn = result_features_knn

        print("Faulty data...")
        faulty_features_dt, faulty_features_knn, faulty_labels_dt, faulty_labels_knn = extract_from_data_sets(
            self.faulty_raw_data, self.window_size, self.features, self.num_outputs)
        self.faulty_labels_dt = faulty_labels_dt
        self.faulty_labels_knn = faulty_labels_knn
        self.faulty_features_dt = faulty_features_dt
        self.faulty_features_knn = faulty_features_knn

    def __glue_routes_together(self, data_set1, data_set2, glue_location):
        pd.set_option('mode.chained_assignment', None)
        set1 = self.__data_sets[data_set1].copy(deep=True)
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
            index = set1.query("cycle > " + str(cycle)).index[0]
            set1.loc[index:, "t_stamp"] = set1.loc[index:, "t_stamp"] + set2_time_between_start_and_end

            # Add routes in correct order into a new data_frame
            route = pd.concat(
                [route, set1_path_of_cycle.loc[:second_glue_item_index], set2_path_of_cycle,
                 set1_path_of_cycle.loc[second_glue_item_index:]], ignore_index=True)

        pd.set_option('mode.chained_assignment', 'warn')
        return route
