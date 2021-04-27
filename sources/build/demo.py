import math
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pandas import DataFrame

from sources.anomaly.topology_guesser import AnomalyTopologyGuesser
from sources.config import BIN_FOLDER_PATH
from sources.data.data_compiler import DataCompiler
from sources.data.features import Features

# from tensorflow import keras

_, is_dt, evaluation_name, simulation_file_path, skip_n = sys.argv
is_dt = int(is_dt) == 1
skip_n = int(skip_n)

WINDOW_SIZE = 35

data = 0
with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_data.pkl", 'rb') as file:
    data = pickle.load(file)

model = 0
if is_dt:
    with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_model.pkl", 'rb') as file:
        model = pickle.load(file)
# else:
#    model = keras.models.load_model(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_model.h5")

model_anomaly = 0
if is_dt:
    with open(BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_dt_anomaly_model.pkl", 'rb') as file:
        model_anomaly = pickle.load(file)
# else:
#    model_anomaly = keras.models.load_model(
#        BIN_FOLDER_PATH + "/" + evaluation_name + "/evaluation_knn_anomaly_model.h5")

# Prepare everything
feature_history = []
x_axis_history = []
y_axis_history = []
df_columns = ["t_stamp", "x_pos", "y_pos", "z_pos", "x_acc", "y_acc", "z_acc", "x_ang", "y_ang", "z_ang", "light",
              "cycle", "pos", "location", "is_anomaly"]
filtered_dataframe = DataFrame(columns=df_columns)
current_feature_index = 0
prediction_history = []
# TP, FP, TN, FN, Not Predicted(?)
statistics_prediction = []
for i in range(len(data.position_map) + 1):
    statistics_prediction.append([0, 0, 0, 0, 0])
statistics_anomaly = [0, 0, 0, 0, 0]
statistics_anomaly_tg = [0, 0, 0, 0, 0]

window_location_changes = []
window_location_changes_no_anomaly = []
window_confidence = []
window_confidence_no_anomaly = []

true_location_history = []
true_anamoly_history = []
anamoly_history = []
anomaly_history_topology_guesser = []

anomaly_areas = [
    [[3, 1], [4.4, 3]]
]

# Prepare plots

fig = plt.figure(figsize=(15, 15))
ax = 0
ax2 = 0
ax3 = 0
ax4 = 0
all_features = []


def redraw():
    global ax
    global ax2
    global ax3
    global ax4
    global all_features
    global data

    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax.set_title("Simulationskarte")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_ylim([-1, 5])
    ax.set_xlim([-1, 5])

    ax2.set_title("Wahrheit vs. Vorhersage: Standort")
    ax2.set_xlabel("Eintrag (Diskret)")
    ax2.set_ylabel("Standort (Diskret)")
    ax2.set_ylim([0, 8])

    ax3.set_title("Wahrheit vs. Vorhersage: Anamolie")
    ax3.set_xlabel("Eintrag (Diskret)")
    ax3.set_ylabel("Ist Anamolie? (Boolean)")
    ax3.set_ylim([-1, 2])

    ax4.set_title("Permutationswichtigkeit im Datenfenster (25)")
    ax4.set_xlabel("Feature")
    ax4.set_ylabel("Fehler in %")

    # Position history
    ax.plot(x_axis_history, y_axis_history, "-b", lw=2, zorder=3)

    size_map = {
        0.1: 180,
        0.2: 500,
        1: 16000,
        2: 60000,
        3: 70000,
        1.5: 33000
    }

    # TODO: Scatter sizes are not linear!
    # Locations:
    ax.scatter([x[0] for x in data.position_map.values()], [x[1] for x in data.position_map.values()], c="green",
               alpha=0.5,
               zorder=2.7, s=size_map[data.proximity])
    for i in range(len(data.position_map.values())):
        ax.text(data.position_map[i + 1][0] - 0.1, data.position_map[i + 1][1] + 0.15, str(i + 1), fontsize=18)

    # Access points
    ax.scatter([x[0] for x in data.access_point_positions], [x[1] for x in data.access_point_positions], c="gray",
               alpha=0.5,
               zorder=2.6, s=size_map[data.access_point_range])
    for i in range(len(data.access_point_positions)):
        ax.text(data.access_point_positions[i][0] - 0.3, data.access_point_positions[i][1] - 0.15, "AP_" + str(i),
                fontsize=18)

    # Heat sources
    ax.scatter([x[0][0] for x in data.heat_sources], [x[0][1] for x in data.heat_sources], c="red", alpha=0.5,
               zorder=2.6, s=[size_map[x[2]] for x in data.heat_sources])
    for i in range(len(data.heat_sources)):
        ax.text(data.heat_sources[i][0][0] - 0.3, data.heat_sources[i][0][1] - 0.15, "HS_" + str(i), fontsize=18)

    # magnetic_sources
    ax.scatter([x[0][0] for x in data.magnetic_sources], [x[0][1] for x in data.magnetic_sources], c="yellow",
               alpha=0.5,
               zorder=2.6, s=[size_map[x[1]] for x in data.magnetic_sources])
    for i in range(len(data.magnetic_sources)):
        ax.text(data.magnetic_sources[i][0][0] - 0.3, data.magnetic_sources[i][0][1] - 0.15, "MS_" + str(i),
                fontsize=18)

    # Noises
    ax.scatter([x[0][0] for x in data.noises], [x[0][1] for x in data.noises], c="purple", alpha=0.5,
               zorder=2.6, s=[size_map[x[2]] for x in data.noises])
    for i in range(len(data.noises)):
        ax.text(data.noises[i][0][0] - 0.3, data.noises[i][0][1] - 0.15, "N_" + str(i), fontsize=18)

    # true vs predicted
    ax2.plot(range(len(true_location_history)), true_location_history, "-g", lw=2, zorder=3)
    ax2.plot(range(len(prediction_history)), prediction_history, "-b", lw=2, zorder=3)

    ax3.plot(range(len(true_anamoly_history)), true_anamoly_history, "-g", lw=2, zorder=3.1)
    ax3.plot(range(len(anamoly_history)), [x - 0.1 for x in anamoly_history], "-b", lw=2, zorder=3)
    ax3.plot(range(len(anomaly_history_topology_guesser)), [x + 0.1 for x in anomaly_history_topology_guesser], "-r", lw=2, zorder=3)

    # Feature importance
    if len(all_features) > 0:
        permutation_importance = model.permutation_importance(all_features, true_location_history[-25:])
        ax4.bar(range(len(permutation_importance)), permutation_importance, align='center')
        plt.xticks(range(len(permutation_importance)), data.name_map_features, size='small', rotation=90)


redraw()

# open the simulation file
file = open(simulation_file_path, "r")

truncated_count = 0

num_draws = 0
now = time.time()
last_prediction = 0
last_prediction_anomaly = 0
last_prediction_anomaly_topology_guesser = 0
last_prediction_when = 0
now = time.time()
num_skipped = 0
prev_distinct_prediction = 0

topology_guesser = AnomalyTopologyGuesser(data.location_neighbor_graph)


def run(args):
    global filtered_dataframe
    global data
    global current_feature_index
    global prediction_history
    global truncated_count
    global now
    global num_draws
    global last_prediction
    global last_prediction_when
    global model
    global model_anomaly
    global window_location_changes
    global window_confidence
    global last_prediction_anomaly
    global statistics_anomaly
    global anomaly_areas
    global ax
    global ax2
    global true_location_history
    global true_anamoly_history
    global anamoly_history
    global all_features
    global skip_n
    global num_skipped
    global window_location_changes_no_anomaly
    global window_confidence_no_anomaly
    global prev_distinct_prediction
    global anomaly_history_topology_guesser
    global last_prediction_anomaly_topology_guesser
    global topology_guesser
    global statistics_anomaly_tg

    # Try to read the line
    current_reader_pos = file.tell()
    line = file.readline()
    if not line:
        file.seek(current_reader_pos)
        return

    if num_skipped < skip_n:
        num_skipped = num_skipped + 1
        return

    # If successful, attempt to parse it
    t_stamp, x_pos, y_pos, z_pos, x_acc, y_acc, z_acc, x_ang, y_ang, z_ang, light, cycle, pos = line.split(",")
    # Ignore the header
    if t_stamp == 't_stamp':
        return

    t_stamp = float(t_stamp)
    x_pos = float(x_pos)
    y_pos = float(y_pos)
    z_pos = float(z_pos)
    x_acc = float(x_acc)
    y_acc = float(y_acc)
    z_acc = float(z_acc)
    x_ang = float(x_ang)
    y_ang = float(y_ang)
    z_ang = float(z_ang)
    light = float(light)
    cycle = int(cycle)
    pos = int(pos)

    x_axis_history.append(x_pos)
    y_axis_history.append(y_pos)

    dataframe = filtered_dataframe.append(
        DataFrame([[t_stamp, x_pos, y_pos, z_pos, x_acc, y_acc, z_acc, x_ang, y_ang, z_ang, light,
                    cycle, pos, 0, False]], columns=df_columns), ignore_index=True)

    # Only keep window_size as feature history
    if len(feature_history) >= data.window_size:
        feature_history.pop(0)

    # Add sensor data
    # Interrupt based filtering
    # Extract the features
    features = [Features.PreviousLocation, Features.AccessPointDetection, Features.Temperature,
                Features.Heading, Features.Volume, Features.Time, Features.Angle, Features.Acceleration, Features.Light]
    processed_data = DataCompiler([], features, False, False, False, 0.1, dataframe, data.num_inputs, data.num_outputs)

    current_features = None
    all_features = []
    cf_count = 0
    for i in range(len(processed_data.result_features_dt[0])):
        all_features = all_features + processed_data.result_features_dt[0][i]
        for j in range(len(processed_data.result_features_dt[0][i])):
            if cf_count == current_feature_index - truncated_count:
                current_features = processed_data.result_features_dt[0][i][j]
                break
            cf_count = cf_count + 1
        if not (current_features is None):
            break

    # Create the filtered dataframe
    filtered_dataframe = dataframe[dataframe.index.isin([0] + processed_data.index_maps[0])]
    if len(filtered_dataframe) > 25:
        truncated_count = truncated_count + (len(filtered_dataframe) - 25)
        filtered_dataframe = filtered_dataframe.tail(25)

    prediction = None
    prediction_proba = None
    if not (current_features is None):
        if current_feature_index > 0:
            prev_prediction = prediction_history[current_feature_index - 1]
            current_features[0] = prev_prediction
            prev_distinct_prediction = 0
            for i in range(len(prediction_history) - 2, 0, -1):
                if prediction_history[i] > 0 and prediction_history[i] != prev_prediction:
                    prev_distinct_prediction = prediction_history[i]
                    break
            current_features[1] = prev_distinct_prediction

        prediction_proba_arr = model.predict_proba([current_features])[0]  # TODO KNN
        prediction = np.asarray(prediction_proba_arr).argmax()
        prediction_proba = prediction_proba_arr[prediction]
        prediction_history.append(prediction)
        current_feature_index = current_feature_index + 1

    # Statistics:
    if not (prediction is None):
        current_location = 0
        for i in range(len(data.position_map.values())):
            if math.sqrt(
                    (data.position_map[i + 1][0] - x_pos) ** 2 + (
                            data.position_map[i + 1][1] - y_pos) ** 2) <= data.proximity:
                current_location = i + 1
                break

        if current_location == prediction:
            statistics_prediction[prediction][0] = statistics_prediction[prediction][0] + 1
        elif current_location != prediction:
            statistics_prediction[prediction][2] = statistics_prediction[prediction][2] + 1
            statistics_prediction[current_location][3] = statistics_prediction[current_location][3] + 1

        for i in range(len(data.position_map.values()) + 1):
            if i != prediction and i != current_location:
                statistics_prediction[i][1] = statistics_prediction[i][1] + 1

        # Anomaly features
        if last_prediction != prediction:
            window_location_changes.append(1)
        else:
            window_location_changes.append(0)

        window_confidence.append(prediction_proba)

        prediction_change = 0 if len(window_confidence) <= 1 else abs(window_confidence[-2] - prediction_proba)
        fraction_zero_prediction = 0
        for pred in prediction_history:
            if pred == 0:
                fraction_zero_prediction = fraction_zero_prediction + 1
        fraction_zero_prediction = fraction_zero_prediction / len(prediction_history)

        if last_prediction_anomaly == 0 and len(window_location_changes) > 1:
            window_location_changes_no_anomaly.append(window_location_changes[-2])
            window_confidence_no_anomaly.append(window_confidence[-2])

        window_loc_changes_deviation = 0
        window_confidence_deviation = 0
        if len(window_location_changes_no_anomaly) >= WINDOW_SIZE:
            window_loc_changes_deviation = abs(
                (sum(window_location_changes_no_anomaly) / max(len(window_location_changes_no_anomaly), 1)) - (
                        sum(window_location_changes[-WINDOW_SIZE:]) / max(len(window_location_changes[-WINDOW_SIZE:]),
                                                                          1)))
            window_confidence_deviation = abs(
                (sum(window_confidence_no_anomaly) / max(len(window_confidence_no_anomaly), 1)) - (
                        sum(window_confidence[-WINDOW_SIZE:]) / max(len(window_confidence[-WINDOW_SIZE:]), 1)))

        anomaly_features = [
            # sum(window_location_changes[-WINDOW_SIZE:]),  # TODO: KNN
            # sum(window_confidence[-WINDOW_SIZE:]),  # TODO: KNN
            # prediction_proba,
            # prediction_change,
            # fraction_zero_prediction,
            window_loc_changes_deviation,
            window_confidence_deviation,
            #prev_distinct_prediction,
            #prediction
        ]

        # Save for statistics
        last_prediction = prediction
        last_prediction_anomaly = model_anomaly.predict([anomaly_features])[0]
        last_prediction_anomaly_topology_guesser = int(
            topology_guesser.predict(prev_distinct_prediction, last_prediction))
        last_prediction_when = t_stamp

        is_current_pos_anomaly = False
        for area in anomaly_areas:
            if area[0][0] <= x_pos <= area[1][0] and area[0][1] <= y_pos <= area[1][1]:
                is_current_pos_anomaly = True
                break

        is_anomaly = last_prediction_anomaly == 1
        if is_anomaly == is_current_pos_anomaly and is_current_pos_anomaly:
            statistics_anomaly[0] = statistics_anomaly[0] + 1
        elif is_anomaly == is_current_pos_anomaly and not is_current_pos_anomaly:
            statistics_anomaly[1] = statistics_anomaly[1] + 1
        elif is_anomaly != is_current_pos_anomaly and is_anomaly:
            statistics_anomaly[2] = statistics_anomaly[2] + 1
        elif is_anomaly != is_current_pos_anomaly and not is_anomaly:
            statistics_anomaly[3] = statistics_anomaly[3] + 1

        is_anomaly_tg = last_prediction_anomaly_topology_guesser == 1
        if is_anomaly_tg == is_current_pos_anomaly and is_current_pos_anomaly:
            statistics_anomaly_tg[0] = statistics_anomaly_tg[0] + 1
        elif is_anomaly_tg == is_current_pos_anomaly and not is_current_pos_anomaly:
            statistics_anomaly_tg[1] = statistics_anomaly_tg[1] + 1
        elif is_anomaly_tg != is_current_pos_anomaly and is_anomaly_tg:
            statistics_anomaly_tg[2] = statistics_anomaly_tg[2] + 1
        elif is_anomaly_tg != is_current_pos_anomaly and not is_anomaly_tg:
            statistics_anomaly_tg[3] = statistics_anomaly_tg[3] + 1

        # History ftw
        true_location_history.append(current_location)
        true_anamoly_history.append(int(is_current_pos_anomaly))
        anamoly_history.append(int(is_anomaly))
        anomaly_history_topology_guesser.append(int(is_anomaly_tg))

    print("Number of predictions: %d" % (len(prediction_history)))
    print("Last Prediction: %d | %.1f seconds ago." % (last_prediction, t_stamp - last_prediction_when))
    print("Is anomaly: %d" % (last_prediction_anomaly))
    print("Is anomaly (TG): %d" % (last_prediction_anomaly_topology_guesser))
    print("Last distinct location %d" % (prev_distinct_prediction))
    print("|   Loc      |   TP   |   TN   |   FP   |   FN   |   ACC  |")
    print("________________________________________________________")
    for i in range(len(data.position_map.values()) + 1):
        sum_row = max(sum(statistics_prediction[i]), 1)
        print("| %d          | %.4f | %.4f | %.4f | %.4f | %.4f |" % (i, (statistics_prediction[i][0] / sum_row),
                                                                      statistics_prediction[i][1] / sum_row,
                                                                      statistics_prediction[i][2] / sum_row,
                                                                      statistics_prediction[i][3] / sum_row,
                                                                      statistics_prediction[i][0] / max(
                                                                          statistics_prediction[i][0] +
                                                                          statistics_prediction[i][3], 1)))

    # Anomaly statistics
    sum_row = max(sum(statistics_anomaly), 1)
    print("| Anomaly    | %.4f | %.4f | %.4f | %.4f | %.4f |" % ((statistics_anomaly[0] / sum_row),
                                                                 statistics_anomaly[1] / sum_row,
                                                                 statistics_anomaly[2] / sum_row,
                                                                 statistics_anomaly[3] / sum_row,
                                                                 (statistics_anomaly[0] + statistics_anomaly[
                                                                     1]) / sum_row))
    sum_row = max(sum(statistics_anomaly_tg), 1)
    print("| Anomaly TG | %.4f | %.4f | %.4f | %.4f | %.4f |" % ((statistics_anomaly_tg[0] / sum_row),
                                                                 statistics_anomaly_tg[1] / sum_row,
                                                                 statistics_anomaly_tg[2] / sum_row,
                                                                 statistics_anomaly_tg[3] / sum_row,
                                                                 (statistics_anomaly_tg[0] + statistics_anomaly_tg[
                                                                     1]) / sum_row))

    # Plot the route
    if num_draws % 25 == 0:
        plt.clf()
        redraw()

    ax.scatter([x_axis_history[-1]], [y_axis_history[-1]], c="blue", s=0.5, zorder=3)

    # Plot real vs discrete position
    if not (prediction is None):
        ax2.scatter([len(true_location_history) - 1], [true_location_history[-1]], c="green", zorder=3)
        ax2.scatter([len(prediction_history) - 1], [prediction_history[-1]], c="blue", zorder=3)

        ax3.scatter([len(true_anamoly_history) - 1], [true_anamoly_history[-1]], c="green", zorder=3.1)
        ax3.scatter([len(anamoly_history) - 1], [anamoly_history[-1] - 0.1], c="blue", zorder=3)
        ax3.scatter([len(anomaly_history_topology_guesser) - 1], [anomaly_history_topology_guesser[-1] + 0.1], c="red", zorder=3)

        # Plot feature importance (using sub plots)
        if prediction > 0:
            ax4.cla()
            ax4.set_title("Permutationswichtigkeit im Datenfenster (25)")
            ax4.set_xlabel("Feature")
            ax4.set_ylabel("Fehler in %")
            permutation_importance = model.permutation_importance(all_features, true_location_history[-25:])
            ax4.bar(range(len(permutation_importance)), permutation_importance, align='center')
            plt.xticks(range(len(permutation_importance)), data.name_map_features, size='small', rotation=90)

    num_draws = num_draws + 1

    print("")
    print("Iteration time: %.2f" % (time.time() - now))
    now = time.time()
    print("---------------------------------------------------------------------")
    print("")


animation = FuncAnimation(plt.gcf(), run, 1, blit=True)
plt.show()
