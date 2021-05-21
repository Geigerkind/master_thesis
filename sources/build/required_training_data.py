import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# from sources.config import BIN_FOLDER_PATH

"""
dt_model_configs = [
    (0, 16, 8, 1, 16, 75, "1"),
    (1, 8, 32, 1, 32, 75, "1"),
    (0, 16, 8, 1, 16, 75, "12"),
    (0, 8, 32, 1, 32, 75, "123"),
    (1, 16, 16, 1, 32, 75, "12"),
    (1, 8, 32, 1, 32, 75, "123"),
    (0, 8, 32, 1, 32, 75, "1234"),
    (1, 16, 16, 1, 32, 75, "1234"),
]

knn_model_configs = [
    (0, 16, 8, 1, 16, 75, "1"),
    (1, 16, 16, 1, 32, 75, "1"),
    (0, 16, 64, 1, 128, 75, "12"),
    (0, 16, 32, 1, 64, 75, "123"),
    (1, 16, 32, 2, 32, 75, "12"),
    (1, 16, 16, 1, 32, 75, "123"),
    (0, 16, 64, 1, 128, 75, "1234"),
    (1, 64, 32, 8, 32, 75, "1234"),
]
"""

BIN_FOLDER_PATH = "/home/shino/Uni/master_thesis/external_eval/bin"
dt_model_configs = [
    (0, 32, 32, 4, 32, 75, "1"),
    (1, 64, 32, 8, 32, 75, "1"),
    (0, 16, 8, 1, 16, 75, "12"),
    (0, 32, 64, 4, 64, 75, "123"),
    (1, 32, 64, 4, 64, 75, "12"),
    (1, 64, 32, 8, 32, 75, "123"),
    (0, 32, 64, 4, 64, 75, "1234"),
    (1, 64, 32, 4, 32, 75, "1234"),
]

knn_model_configs = [
    (0, 64, 32, 8, 32, 75, "1"),
    (1, 16, 32, 2, 32, 75, "1"),
    (0, 16, 64, 1, 128, 75, "12"),
    (0, 64, 32, 8, 32, 75, "123"),
    (1, 32, 64, 4, 64, 75, "12"),
    (1, 16, 64, 1, 128, 75, "123"),
    (0, 16, 32, 1, 64, 75, "1234"),
    (1, 16, 32, 1, 64, 75, "1234"),
]

location_map = [9, 16, 17, 25, 32, 48, 52, 102]

result_dt = []
result_knn = []

LF_ACC = 0.97

for i in range(len(dt_model_configs)):
    dt_cycles = 16
    knn_cycles = 16
    evaluation_name_dt = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(dt_model_configs[i][0],
                                                                             dt_model_configs[i][1],
                                                                             dt_model_configs[i][2],
                                                                             dt_model_configs[i][3],
                                                                             dt_model_configs[i][4],
                                                                             dt_model_configs[i][5],
                                                                             dt_model_configs[i][6])

    dt_accs = pd.read_csv(BIN_FOLDER_PATH + "/" + evaluation_name_dt + "/log_accuracy_per_cycle.csv")

    evaluation_name_knn = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(knn_model_configs[i][0],
                                                                              knn_model_configs[i][1],
                                                                              knn_model_configs[i][2],
                                                                              knn_model_configs[i][3],
                                                                              knn_model_configs[i][4],
                                                                              knn_model_configs[i][5],
                                                                              knn_model_configs[i][6])

    knn_accs = pd.read_csv(BIN_FOLDER_PATH + "/" + evaluation_name_knn + "/log_accuracy_per_cycle.csv")

    for row in dt_accs.iterrows():
        if row[1]["accuracy_dt_test"] >= LF_ACC:
            dt_cycles = row[1]["cycle"] + 1
            break

    for row in knn_accs.iterrows():
        if row[1]["accuracy_knn_test"] >= LF_ACC:
            knn_cycles = row[1]["cycle"] + 1
            break

    result_dt.append(dt_cycles)
    result_knn.append(knn_cycles)

fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
plt.bar(np.arange(len(location_map)) - 0.2, result_dt, 0.4, label="Entscheidungswald")
plt.bar(np.arange(len(location_map)) + 0.2, result_knn, 0.4, label="FFNN")
plt.xticks(np.arange(len(location_map)), location_map)
plt.xlabel("Standortkomplexität")
plt.ylabel("Anzahl der Trainingszyklen")
plt.legend()
plt.savefig("{0}/required_training_data.png".format(BIN_FOLDER_PATH))
plt.clf()
plt.close(fig)