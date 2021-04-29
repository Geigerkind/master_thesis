import locale
import os
import re

import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame


class CompileAll:
    def __init__(self, bin_path):
        # Parameters
        self.bin_path = bin_path

        # Internal helper variables
        self.distinct_number_of_locations = [9, 16, 17, 25, 32, 48, 52, 102]
        acc_types = ["acc", "acc_pc", "acc_5", "acc_10", "acc_cont", "acc_pc_cont", "acc_5_cont", "acc_10_cont"]

        # Here is where the action happens
        locale.setlocale(locale.LC_ALL, 'de_DE')
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

    def __load_log_accuracies(self):
        def extract_accuracies_from_file(file):
            data_set = pandas.read_csv(file)
            return data_set.iloc[0]["accuracy"], data_set.iloc[0]["accuracy_given_previous_location_was_correct"], \
                   data_set.iloc[0]["accuracy_given_location_is_cont_the_same_and_within_5_entries"], \
                   data_set.iloc[0]["accuracy_given_location_is_cont_the_same_and_within_10_entries"]

        data = []
        bin_path_len = len(self.bin_path)
        regex_template = re.compile("\d+")
        for subdir, dirs, files in os.walk(self.bin_path):
            subdir_len = len(subdir)
            if "bin/eval" in subdir:
                re_match = regex_template.findall(subdir[bin_path_len:])
                path_encoded = int(re_match[0]) == 1
                num_trees = int(re_match[1])
                max_depth = int(re_match[2])
                num_layers = int(re_match[3])
                num_neurons = int(re_match[4])
                data_sets = int(re_match[5])

                num_locations = 0
                if data_sets == 1:
                    num_locations = 8
                elif data_sets == 12:
                    num_locations = 16
                elif data_sets == 123:
                    num_locations = 24
                elif data_sets == 124:
                    num_locations = 51

                if path_encoded:
                    num_locations = num_locations * 2
                else:
                    num_locations = num_locations + 1

                for dir in dirs:
                    is_faulty = "faulty" in dir
                    is_test_data = "_test" in dir

                    if not (is_faulty or is_test_data):
                        continue

                    route = dir[subdir_len + 8:] if is_faulty else dir[subdir_len + 1]

                    dt_acc, dt_acc_pc, dt_acc_5, dt_acc_10 = extract_accuracies_from_file(
                        subdir + "/" + dir + "/evaluation_dt/log_true_vs_predicted.csv")

                    dt_acc_cont, dt_acc_pc_cont, dt_acc_5_cont, dt_acc_10_cont = extract_accuracies_from_file(
                        subdir + "/" + dir + "/evaluation_continued_dt/log_true_vs_predicted.csv")

                    knn_acc, knn_acc_pc, knn_acc_5, knn_acc_10 = extract_accuracies_from_file(
                        subdir + "/" + dir + "/evaluation_knn/log_true_vs_predicted.csv")

                    knn_acc_cont, knn_acc_pc_cont, knn_acc_5_cont, knn_acc_10_cont = extract_accuracies_from_file(
                        subdir + "/" + dir + "/evaluation_continued_knn/log_true_vs_predicted.csv")

                    data.append([route, is_faulty, num_trees, max_depth, num_layers, num_neurons, num_locations, dt_acc,
                                 dt_acc_pc, dt_acc_5, dt_acc_10, dt_acc_cont, dt_acc_pc_cont, dt_acc_5_cont,
                                 dt_acc_10_cont, knn_acc, knn_acc_pc, knn_acc_5, knn_acc_10, knn_acc_cont,
                                 knn_acc_pc_cont, knn_acc_5_cont, knn_acc_10_cont])

        return DataFrame(data, columns=["route", "is_faulty",
                                        "trees", "max_depth", "layers", "neurons", "num_locations",
                                        "dt_acc", "dt_acc_pc", "dt_acc_5", "dt_acc_10",
                                        "dt_acc_cont", "dt_acc_pc_cont", "dt_acc_5_cont", "dt_acc_10_cont",
                                        "knn_acc", "knn_acc_pc", "knn_acc_5", "knn_acc_10",
                                        "knn_acc_cont", "knn_acc_pc_cont", "knn_acc_5_cont", "knn_acc_10_cont"])

    def __generate_graph_best_dt_vs_best_ffnn(self, acc_kind):
        dt_accs = []
        knn_accs = []
        for num_loc in self.distinct_number_of_locations:
            by_loc = self.prediction_accuracies.query("not is_faulty and num_locations == " + str(num_loc))
            accs_per_dt = dict()
            accs_per_knn = dict()
            for row in by_loc.iterrows():
                key_dt = (row[1]["num_trees"], row[1]["max_depth"])
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

        plt.figure(figsize=(15 / 2.54, 30 / 2.54))
        fig, ax1 = plt.subplots()
        ax1.plot(self.prediction_accuracies, dt_accs, "o-g")
        ax1.plot(self.prediction_accuracies, knn_accs, "*-b")
        ax1.set_xlabel("Anzahl Standorte (Diskret)")
        ax1.set_ylabel("Klassifizierungsgenauigkeit")
        ax1.set_ylim([0, 1])
        fig.legend(['Entscheidungsbaum', 'FFNN'], loc=[0.68, 0.77])
        plt.savefig("{0}/best_dt_vs_best_ffnn_over_num_loc_using_{1}.png".format(self.bin_path, acc_kind))
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
                    accs.append(value)
                else:
                    accs[entry] = [value]

        plt.figure(figsize=(15 / 2.54, 30 / 2.54))
        fig, ax1 = plt.subplots()
        for group in possible_groups:
            ax1.plot(self.distinct_number_of_locations, accs[group])
        ax1.set_xlabel("Anzahl Standorte (Diskret)")
        ax1.set_ylabel("Klassifizierungsgenauigkeit")
        ax1.set_ylim([0, 1])
        fig.legend([label_function(x) for x in possible_groups], loc=[0.68, 0.77])
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
                key_dt = (row[1]["num_trees"], row[1]["max_depth"])
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

        group_order1 = []
        group_order2 = []

        file = open(self.bin_path + "/predictions_by_" + acc_kind + ".tex")
        file.write("\\begin{table}[h!]\n")
        file.write("\\begin{tabular}{ | c | c | c | c | c | c | c | c | c | c | }\n")
        file.write(
            "\\multicolumn{2}{ | l |}{" + acc_kind + " / Standorte } & 9 & 16 & 17 & 25 & 32 & 48 & 52 & 102 \\\\\\hline\n")
        file.write("\\multicolumn{9}{| l |}{\\textbf{Entscheidungswälder}}\\\\\\hline\n")
        file.write("Waldgröße & Max. Baumgröße & \\multicolumn{7}{| c |}{}\\\\\\hline\n")
        for group in group_order1:
            file.write("{0} & {1} & {2:.2f}% & {3:.2f}% & {4:.2f}% & {5:.2f}% & {6:.2f}% & {7:.2f}% & {8:.2f}% "
                       "& {9:.2f}% \\\\\\hline\n".format(group[0], group[1], dt_accs[group][0],
                                                                     dt_accs[group][1], dt_accs[group][2],
                                                                     dt_accs[group][3], dt_accs[group][4],
                                                                     dt_accs[group][5], dt_accs[group][6],
                                                                     dt_accs[group][7]))
        file.write("\\multicolumn{9}{| l |}{\\textbf{Feed Forward neuronale Netzwerke}}\\\\\\hline\n")
        file.write("#Schichten & #Neuronen & \\multicolumn{7}{| c |}{}\\\\\\hline\n")
        for group in group_order2:
            file.write("{0} & {1} & {2:.2f}% & {3:.2f}% & {4:.2f}% & {5:.2f}% & {6:.2f}% & {7:.2f}% & {8:.2f}% "
                       "& {9:.2f}% \\\\\\hline\n".format(group[0], group[1], knn_accs[group][0],
                                                         knn_accs[group][1], knn_accs[group][2],
                                                         knn_accs[group][3], knn_accs[group][4],
                                                         knn_accs[group][5], knn_accs[group][6],
                                                         knn_accs[group][7]))
        file.write("\\end{tabular}\n")
        file.write("\\caption{TODO: Caption}\n")
        file.write("\\label{tab:TODO_LABEL}\n")
        file.write("\\end{table}\n")


CompileAll("/home/shino/Uni/master_thesis/bin")
