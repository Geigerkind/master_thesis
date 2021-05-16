import locale
import os
import pickle
import re

import matplotlib as mpl

from sources.config import WITHOUT_PREVIOUS_EDGE

mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame


class CompileAll:
    def __init__(self, bin_path):
        # Parameters
        self.bin_path = bin_path

        # Internal helper variables
        self.distinct_number_of_locations = [9, 16, 17, 25, 32, 48, 52, 102]
        acc_types = ["acc", "acc_pc", "acc_pic", "acc_5", "acc_10", "acc_cont", "acc_pc_cont", "acc_pic_cont",
                     "acc_5_cont", "acc_10_cont"]

        # Here is where the action happens
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
        self.prediction_accuracies = self.__load_log_accuracies()
        self.prediction_accuracies.to_csv(self.bin_path + "/log_compiled.csv")

        for acc_type in acc_types:
            # Graphs
            self.__generate_graph_best_dt_vs_best_ffnn(acc_type)
            self.__generate_graph_multiple_best_ml_group_by("dt", "trees", [8, 16, 32, 64], "max_depth == 32", acc_type,
                                                            lambda trees: "Waldgröße: {0}".format(trees))
            self.__generate_graph_multiple_best_ml_group_by("dt", "max_depth", [8, 16, 32, 64], "trees == 16", acc_type,
                                                            lambda trees: "Max. Baumgröße: {0}".format(trees))
            self.__generate_graph_multiple_best_ml_group_by("knn", "neurons", [16, 32, 64, 128], "layers == 1",
                                                            acc_type, lambda trees: "#Neuronen: {0}".format(trees))
            self.__generate_graph_multiple_best_ml_group_by("knn", "layers", [1, 2, 4, 8], "neurons == 32",
                                                            acc_type, lambda trees: "#Schichten: {0}".format(trees))

            # Latex tables
            self.__generate_latex_table(acc_type)
        self.__generate_latex_table("loc_size")
        self.__generate_graph_best_dt_vs_best_ffnn_vs_kind("acc", "acc_cont")
        self.__generate_faulty_latex_table("acc_cont")
        self.__generate_faulty_avg_over_trees("acc_cont")

    def __load_log_accuracies(self):
        def extract_accuracies_from_file(file):
            data_set = pandas.read_csv(file)
            return data_set.iloc[0]["accuracy"], data_set.iloc[0]["accuracy_given_previous_location_was_correct"], \
                   data_set.iloc[0]["accuracy_given_previous_location_was_incorrect"], \
                   data_set.iloc[0]["accuracy_given_location_is_cont_the_same_and_within_5_entries"], \
                   data_set.iloc[0]["accuracy_given_location_is_cont_the_same_and_within_10_entries"]

        data = []
        bin_path_len = len(self.bin_path)
        regex_template = re.compile("\d+")
        size_map = dict()
        for subdir, dirs, files in os.walk(self.bin_path):
            subdir_len = len(subdir)
            if "DS_1" == subdir[-4:] or "DS_12" == subdir[-5:] or "DS_123" == subdir[-6:] or "DS_1234" == subdir[-7:]:
                continue

            if "/evaluation" in subdir or "combined" in subdir or "anomaly" in subdir:
                continue

            if "bin/eval" in subdir:
                re_match = regex_template.findall(subdir[bin_path_len:])
                path_encoded = int(re_match[0]) == 1
                num_trees = int(re_match[1])
                max_depth = int(re_match[2])
                num_layers = int(re_match[3])
                num_neurons = int(re_match[4])
                data_sets = int(re_match[6])

                num_locations = 0
                if data_sets == 1:
                    num_locations = 8
                elif data_sets == 12:
                    num_locations = 16
                elif data_sets == 123:
                    num_locations = 24
                elif data_sets == 1234:
                    num_locations = 51

                if path_encoded:
                    num_locations = num_locations * 2
                else:
                    num_locations = num_locations + 1

                is_faulty = "faulty" in subdir
                is_test_data = "_test" in subdir

                if not (is_faulty or is_test_data):
                    continue

                slash_index = list(reversed(subdir)).index("/")
                route = subdir[-slash_index + 7:] if is_faulty else subdir[-slash_index:]

                dt_acc, dt_acc_pc, dt_acc_pic, dt_acc_5, dt_acc_10 = extract_accuracies_from_file(
                    subdir + "/evaluation_dt/log_true_vs_predicted.csv")

                dt_acc_cont, dt_acc_pc_cont, dt_acc_pic_cont, dt_acc_5_cont, dt_acc_10_cont = (0, 0, 0, 0, 0)
                if not WITHOUT_PREVIOUS_EDGE:
                    dt_acc_cont, dt_acc_pc_cont, dt_acc_pic_cont, dt_acc_5_cont, dt_acc_10_cont = extract_accuracies_from_file(
                        subdir + "/evaluation_continued_dt/log_true_vs_predicted.csv")

                knn_acc, knn_acc_pc, knn_acc_pic, knn_acc_5, knn_acc_10 = extract_accuracies_from_file(
                    subdir + "/evaluation_knn/log_true_vs_predicted.csv")

                knn_acc_cont, knn_acc_pc_cont, knn_acc_pic_cont, knn_acc_5_cont, knn_acc_10_cont = (0, 0, 0, 0, 0)
                if not WITHOUT_PREVIOUS_EDGE:
                    knn_acc_cont, knn_acc_pc_cont, knn_acc_pic_cont, knn_acc_5_cont, knn_acc_10_cont = extract_accuracies_from_file(
                        subdir + "/evaluation_continued_knn/log_true_vs_predicted.csv")

                # This could be taken out of the loop
                #loc_real_max_depth, loc_size_dt, loc_size_knn, anomaly_real_max_depth, \
                #anomaly_size_dt, anomaly_size_knn = [0, 0, 0, 0, 0, 0]
                size_key = ((subdir[::-1])[slash_index + 1:])[::-1]
                if size_key in size_map:
                    loc_real_max_depth, loc_size_dt, loc_size_knn, anomaly_real_max_depth, \
                    anomaly_size_dt, anomaly_size_knn = size_map.get(size_key)
                else:
                    file_dt_loc = open(size_key + "/evaluation_dt_model.pkl", "rb")
                    file_dt_anomaly = open(size_key + "/evaluation_dt_anomaly_model.pkl", "rb")
                    model_loc_dt = pickle.load(file_dt_loc)
                    model_anomaly_dt = pickle.load(file_dt_anomaly)

                    loc_real_max_depth = model_loc_dt.max_depth_forest()
                    loc_size_dt = model_loc_dt.calculate_size()
                    loc_size_knn = num_neurons * (34 + num_locations + (num_layers - 1) * num_neurons) * 4

                    anomaly_real_max_depth = model_anomaly_dt.max_depth_forest()
                    anomaly_size_dt = model_anomaly_dt.calculate_size()
                    anomaly_size_knn = 32 * 35 * 4

                    file_dt_loc.close()
                    file_dt_anomaly.close()
                    size_map[size_key] = [loc_real_max_depth, loc_size_dt, loc_size_knn, anomaly_real_max_depth,
                                          anomaly_size_dt, anomaly_size_knn]

                data.append([route, is_faulty, num_trees, max_depth, num_layers, num_neurons, num_locations, dt_acc,
                             dt_acc_pc, dt_acc_pic, dt_acc_5, dt_acc_10, dt_acc_cont, dt_acc_pc_cont, dt_acc_pic_cont,
                             dt_acc_5_cont, dt_acc_10_cont, knn_acc, knn_acc_pc, knn_acc_pic, knn_acc_5, knn_acc_10,
                             knn_acc_cont, knn_acc_pc_cont, knn_acc_pic_cont, knn_acc_5_cont, knn_acc_10_cont,
                             loc_real_max_depth, loc_size_dt, loc_size_knn, anomaly_real_max_depth, anomaly_size_dt,
                             anomaly_size_knn])

        return DataFrame(data, columns=["route", "is_faulty",
                                        "trees", "max_depth", "layers", "neurons", "num_locations",
                                        "dt_acc", "dt_acc_pc", "dt_acc_pic", "dt_acc_5", "dt_acc_10",
                                        "dt_acc_cont", "dt_acc_pc_cont", "dt_acc_pic_cont", "dt_acc_5_cont",
                                        "dt_acc_10_cont",
                                        "knn_acc", "knn_acc_pc", "knn_acc_pic", "knn_acc_5", "knn_acc_10",
                                        "knn_acc_cont", "knn_acc_pc_cont", "knn_acc_pic_cont", "knn_acc_5_cont",
                                        "knn_acc_10_cont", "loc_real_max_depth", "dt_loc_size", "knn_loc_size",
                                        "anomaly_real_max_depth", "dt_anomaly_size", "knn_anomaly_size"])

    def __generate_graph_best_dt_vs_best_ffnn(self, acc_kind):
        dt_accs = []
        knn_accs = []
        for num_loc in self.distinct_number_of_locations:
            by_loc = self.prediction_accuracies.query("not is_faulty and num_locations == " + str(num_loc))
            accs_per_dt = dict()
            accs_per_knn = dict()
            for row in by_loc.iterrows():
                key_dt = (row[1]["trees"], row[1]["max_depth"])
                if key_dt in accs_per_dt:
                    accs_per_dt[key_dt].append(row[1]["dt_" + acc_kind])
                else:
                    accs_per_dt[key_dt] = [row[1]["dt_" + acc_kind]]

                key_knn = (row[1]["layers"], row[1]["neurons"])
                if key_knn in accs_per_knn:
                    accs_per_knn[key_knn].append(row[1]["knn_" + acc_kind])
                else:
                    accs_per_knn[key_knn] = [row[1]["knn_" + acc_kind]]

            max_config_dt = max(accs_per_dt, key=lambda i: sum(accs_per_dt[i]))
            dt_accs.append(sum(accs_per_dt[max_config_dt]) / len(accs_per_dt[max_config_dt]))
            max_config_knn = max(accs_per_knn, key=lambda i: sum(accs_per_knn[i]))
            knn_accs.append(sum(accs_per_knn[max_config_knn]) / len(accs_per_knn[max_config_knn]))

        fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
        plt.plot(self.distinct_number_of_locations, dt_accs, "o-g")
        plt.plot(self.distinct_number_of_locations, knn_accs, "*-b")
        plt.xlabel("Anzahl Standorte (Diskret)")
        plt.ylabel("Klassifizierungsgenauigkeit")
        plt.ylim([0, 1])
        fig.legend(['Entscheidungsbaum', 'FFNN'], loc=[0.13, 0.13])
        plt.savefig("{0}/best_dt_vs_best_ffnn_over_num_loc_using_{1}.png".format(self.bin_path, acc_kind))
        plt.clf()
        plt.close(fig)

    def __generate_graph_best_dt_vs_best_ffnn_vs_kind(self, acc_kind, vs_acc_kind):
        dt_accs = []
        dt_accs_vs = []
        knn_accs = []
        knn_accs_vs = []
        for num_loc in self.distinct_number_of_locations:
            by_loc = self.prediction_accuracies.query("not is_faulty and num_locations == " + str(num_loc))
            accs_per_dt = dict()
            accs_per_dt_vs = dict()
            accs_per_knn = dict()
            accs_per_knn_vs = dict()
            for row in by_loc.iterrows():
                key_dt = (row[1]["trees"], row[1]["max_depth"])
                if key_dt in accs_per_dt:
                    accs_per_dt[key_dt].append(row[1]["dt_" + acc_kind])
                else:
                    accs_per_dt[key_dt] = [row[1]["dt_" + acc_kind]]

                if key_dt in accs_per_dt_vs:
                    accs_per_dt_vs[key_dt].append(row[1]["dt_" + vs_acc_kind])
                else:
                    accs_per_dt_vs[key_dt] = [row[1]["dt_" + vs_acc_kind]]

                key_knn = (row[1]["layers"], row[1]["neurons"])
                if key_knn in accs_per_knn:
                    accs_per_knn[key_knn].append(row[1]["knn_" + acc_kind])
                else:
                    accs_per_knn[key_knn] = [row[1]["knn_" + acc_kind]]
                if key_knn in accs_per_knn_vs:
                    accs_per_knn_vs[key_knn].append(row[1]["knn_" + vs_acc_kind])
                else:
                    accs_per_knn_vs[key_knn] = [row[1]["knn_" + vs_acc_kind]]

            max_config_dt = max(accs_per_dt, key=lambda i: sum(accs_per_dt[i]))
            dt_accs.append(sum(accs_per_dt[max_config_dt]) / len(accs_per_dt[max_config_dt]))
            max_config_knn = max(accs_per_knn, key=lambda i: sum(accs_per_knn[i]))
            knn_accs.append(sum(accs_per_knn[max_config_knn]) / len(accs_per_knn[max_config_knn]))

            max_config_dt_vs = max(accs_per_dt_vs, key=lambda i: sum(accs_per_dt_vs[i]))
            dt_accs_vs.append(sum(accs_per_dt_vs[max_config_dt_vs]) / len(accs_per_dt_vs[max_config_dt_vs]))
            max_config_knn_vs = max(accs_per_knn_vs, key=lambda i: sum(accs_per_knn_vs[i]))
            knn_accs_vs.append(sum(accs_per_knn_vs[max_config_knn_vs]) / len(accs_per_knn_vs[max_config_knn_vs]))

        fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
        plt.plot(self.distinct_number_of_locations, dt_accs, "o-g")
        plt.plot(self.distinct_number_of_locations, dt_accs_vs, "*-g")
        plt.plot(self.distinct_number_of_locations, knn_accs, "o-b")
        plt.plot(self.distinct_number_of_locations, knn_accs_vs, "*-b")
        plt.xlabel("Anzahl Standorte (Diskret)")
        plt.ylabel("Klassifizierungsgenauigkeit")
        plt.ylim([0, 1])
        fig.legend(['Entscheidungsbaum P(A)', 'Entscheidungsbaum P(A) (cont.)', 'FFNN P(A)', 'FFNN P(A) (cont.)'],
                   loc=[0.13, 0.13])
        plt.savefig("{0}/best_dt_vs_knn_{1}_vs_{2}.png".format(self.bin_path, acc_kind, vs_acc_kind))
        plt.clf()
        plt.close(fig)

    def __generate_graph_multiple_best_ml_group_by(self, ml_kind, grouping_key, possible_groups, filter_query, acc_kind,
                                                   label_function):
        accs = dict()
        for num_loc in self.distinct_number_of_locations:
            by_loc = self.prediction_accuracies.query("not is_faulty and num_locations == " + str(num_loc) +
                                                      " and " + filter_query)
            accs_per = dict()
            for row in by_loc.iterrows():
                key = row[1][grouping_key]
                if key in accs_per:
                    accs_per[key].append(row[1][ml_kind + "_" + acc_kind])
                else:
                    accs_per[key] = [row[1][ml_kind + "_" + acc_kind]]

            for entry in accs_per.keys():
                value = sum(accs_per[entry]) / len(accs_per[entry])
                if entry in accs:
                    accs[entry].append(value)
                else:
                    accs[entry] = [value]

        fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
        for group in possible_groups:
            plt.plot(self.distinct_number_of_locations, accs[group])
        plt.xlabel("Anzahl Standorte (Diskret)")
        plt.ylabel("Klassifizierungsgenauigkeit")
        plt.ylim([0, 1])
        fig.legend([label_function(x) for x in possible_groups], loc=[0.13, 0.13])
        plt.savefig("{0}/multiple_best_by_group_{1}_{2}_{3}.png".format(self.bin_path, ml_kind, grouping_key, acc_kind))
        plt.clf()
        plt.close(fig)

    def __generate_latex_table(self, acc_kind):
        dt_accs = dict()
        knn_accs = dict()
        for num_loc in self.distinct_number_of_locations:
            by_loc = self.prediction_accuracies.query("not is_faulty and num_locations == " + str(num_loc))
            accs_per_dt = dict()
            accs_per_knn = dict()
            for row in by_loc.iterrows():
                key_dt = (row[1]["trees"], row[1]["max_depth"])
                if key_dt in accs_per_dt:
                    accs_per_dt[key_dt].append(row[1]["dt_" + acc_kind])
                else:
                    accs_per_dt[key_dt] = [row[1]["dt_" + acc_kind]]

                key_knn = (row[1]["layers"], row[1]["neurons"])
                if key_knn in accs_per_knn:
                    accs_per_knn[key_knn].append(row[1]["knn_" + acc_kind])
                else:
                    accs_per_knn[key_knn] = [row[1]["knn_" + acc_kind]]

            for key in accs_per_dt.keys():
                value = sum(accs_per_dt[key]) / len(accs_per_dt[key])
                if key in dt_accs:
                    dt_accs[key].append(value)
                else:
                    dt_accs[key] = [value]

            for key in accs_per_knn.keys():
                value = sum(accs_per_knn[key]) / len(accs_per_knn[key])
                if key in knn_accs:
                    knn_accs[key].append(value)
                else:
                    knn_accs[key] = [value]

        group_order1 = [(16, 8), (16, 16), (16, 32), (16, 64), (8, 32), (32, 32), (64, 32), (32, 64)]
        group_order2 = [(1, 16), (1, 32), (1, 64), (1, 128), (2, 32), (4, 32), (8, 32), (4, 64)]
        label_map = {
            "acc": "$P(A)$",
            "acc_5": "$P(B=5)$",
            "acc_10": "$P(B=10)$",
            "acc_pc": "$P(C)$",
            "acc_pic": "$P(D)$",
            "acc_cont": "$P(A)_{\\text{cont}}$",
            "acc_5_cont": "$P(B=5)_{\\text{cont}}$",
            "acc_10_cont": "$P(B=10)_{\\text{cont}}$",
            "acc_pc_cont": "$P(C)_{\\text{cont}}$",
            "acc_pic_cont": "$P(D)_{\\text{cont}}$",
            "loc_size": "Programmgröße in KB",
            "anomaly_size": "Programmgröße in KB",
        }
        acc_kind_factor = 0.001 if acc_kind == "loc_size" else 100
        row_format = "{0} & {1} & {2:.1f} & {3:.1f} & {4:.1f} & {5:.1f} & {6:.1f} & {7:.1f} & {8:.1f} & {9:.1f} \\\\\\hline\n" if acc_kind == "loc_size" else "{0} & {1} & {2:.2f}\\% & {3:.2f}\\% & {4:.2f}\\% & {5:.2f}\\% & {6:.2f}\\% & {7:.2f}\\% & {8:.2f}\\% & {9:.2f}\\% \\\\\\hline\n"

        file = open(self.bin_path + "/predictions_by_" + acc_kind + ".tex", "w")
        file.write("\\begin{table}[h!]\n")
        file.write("\\hspace{-2cm}\n")
        file.write("\\begin{tabular}{ | c | c | c | c | c | c | c | c | c | c | }\n")
        file.write("\\hline\n")
        file.write(
            "\\multicolumn{2}{ | l |}{" + label_map.get(
                acc_kind) + " über Standorte} & 9 & 16 & 17 & 25 & 32 & 48 & 52 & 102 \\\\\\hline\n")
        file.write("\\multicolumn{10}{| l |}{\\textbf{Entscheidungswälder}}\\\\\\hline\n")
        file.write("Waldgröße & Max. Baumgröße & \\multicolumn{8}{ c |}{}\\\\\\hline\n")
        for group in group_order1:
            file.write(row_format.format(group[0], group[1], acc_kind_factor * dt_accs[group][0],
                                         acc_kind_factor * dt_accs[group][1],
                                         acc_kind_factor * dt_accs[group][2],
                                         acc_kind_factor * dt_accs[group][3],
                                         acc_kind_factor * dt_accs[group][4],
                                         acc_kind_factor * dt_accs[group][5],
                                         acc_kind_factor * dt_accs[group][6],
                                         acc_kind_factor * dt_accs[group][7]))
        file.write("\\multicolumn{10}{| l |}{\\textbf{Feed Forward neuronale Netzwerke}}\\\\\\hline\n")
        file.write("\\#Schichten & \\#Neuronen & \\multicolumn{8}{ c |}{}\\\\\\hline\n")
        for group in group_order2:
            file.write(row_format.format(group[0], group[1], acc_kind_factor * knn_accs[group][0],
                                         acc_kind_factor * knn_accs[group][1],
                                         acc_kind_factor * knn_accs[group][2],
                                         acc_kind_factor * knn_accs[group][3],
                                         acc_kind_factor * knn_accs[group][4],
                                         acc_kind_factor * knn_accs[group][5],
                                         acc_kind_factor * knn_accs[group][6],
                                         acc_kind_factor * knn_accs[group][7]))
        file.write("\\end{tabular}\n")
        file.write("\\caption{Metrik " + label_map.get(
            acc_kind) + " über Standorte und verschiedenen Konfigurationen der ML-Modelle.}\n")
        file.write("\\label{tab:predictions_by_" + acc_kind + "}\n")
        file.write("\\end{table}\n")

    def __generate_faulty_latex_table(self, acc_kind):
        location_complexity = 9
        best_dt = (64, 32)
        best_knn = (1, 128)

        dt_data = self.prediction_accuracies.query(
            "trees == " + str(best_dt[0]) + " and max_depth == " + str(best_dt[1]) + " and num_locations == " + str(
                location_complexity))

        knn_data = self.prediction_accuracies.query(
            "layers == " + str(best_knn[0]) + " and neurons == " + str(best_knn[1]) + " and num_locations == " + str(
                location_complexity))

        dt_base_acc = dt_data.query("route == 'simple_square_test'").iloc[0]["dt_" + acc_kind]
        knn_base_acc = knn_data.query("route == 'simple_square_test'").iloc[0]["knn_" + acc_kind]
        dt_data = dt_data.query("is_faulty == True")
        knn_data = knn_data.query("is_faulty == True")

        route_accuracies = dict()
        for row in dt_data.iterrows():
            route_accuracies[row[1]["route"]] = (row[1]["dt_" + acc_kind] - dt_base_acc, 0)
        for row in knn_data.iterrows():
            route_accuracies[row[1]["route"]] = (route_accuracies[row[1]["route"]][0], row[1]["knn_" + acc_kind] - knn_base_acc)

        file = open(self.bin_path + "/robustness_by_" + acc_kind + ".tex", "w")
        file.write("\\begin{table}[h!]\n")
        file.write("\\begin{tabular}{ | l | c | c | }\n")
        file.write("\\hline\n")
        file.write("Testmenge & Entscheidungswald & FFNN \\\\\\hline\n")
        for route in route_accuracies:
            file.write("{0} & {1:0.2f}\% & {2:0.2f}\% \\\\\\hline\n".format(route, 100 * route_accuracies[route][0], 100 * route_accuracies[route][1]))
        file.write("\\end{tabular}\n")
        file.write("\\caption{TODO}\n")
        file.write("\\label{tab:robustness_by_" + acc_kind + "}\n")
        file.write("\\end{table}\n")

    def __generate_faulty_avg_over_trees(self, acc_kind):
        dt_data = self.prediction_accuracies.query("max_depth == 32 and num_locations == 9")
        dt_data_faulty = dt_data.query("is_faulty == True")

        dt_accs = dict()
        dt_count = dict()

        for row in dt_data_faulty.iterrows():
            if row[1]["route"] == "simple_square_test_skipped_location" or ('random' in row[1]["route"]):
                continue

            base_acc = dt_data.query("route == 'simple_square_test' and trees == " + str(row[1]["trees"])).iloc[0]["dt_" + acc_kind]
            if row[1]["trees"] in dt_accs:
                dt_accs[row[1]["trees"]] = dt_accs[row[1]["trees"]] + (row[1]["dt_" + acc_kind] - base_acc)
                dt_count[row[1]["trees"]] = dt_count[row[1]["trees"]] + 1
            else:
                dt_accs[row[1]["trees"]] = row[1]["dt_" + acc_kind] - base_acc
                dt_count[row[1]["trees"]] = 1

        avg_accs = []
        trees = []
        for num_trees in dt_accs:
            trees.append(num_trees)
            avg_accs.append(dt_accs[num_trees] / dt_count[num_trees])

        print(avg_accs)
        print(trees)

CompileAll("/home/shino/Uni/master_thesis/bin")
