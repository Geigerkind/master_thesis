import pandas as pd


class CompileLog:
    def __init__(self, path, folder_prefix):
        self.folder_prefix = folder_prefix

        file_path = path + folder_prefix
        log_true_vs_predicted = pd.read_csv(file_path + "/log_true_vs_predicted.csv")
        log_recognized_path_segment = pd.read_csv(file_path + "/log_recognized_path_segment.csv")
        log_path_segment_misclassified = pd.read_csv(file_path + "/log_path_segment_misclassified.csv")
        log_location_misclassified = pd.read_csv(file_path + "/log_location_misclassified.csv")
        log_location_misclassification = pd.read_csv(file_path + "/log_location_misclassification.csv")

        log_compiled = open(path + "evaluation/log_compiled.csv", "w")
        log_compiled.write("accuracy,accuracy_given_previous_location_was_correct,"
                           "accuracy_given_previous_location_was_incorrect,"
                           "accuracy_given_location_is_cont_the_same_and_within_5_entries,"
                           "accuracy_given_location_is_cont_the_same_and_within_10_entries,"
                           "average_path_recognition_delay,times_not_found_path\n")
        log_compiled.close()

        log_compiled = open(path + "evaluation/log_compiled_location.csv", "w")
        log_compiled.write("location,times_misclassified_as,times_misclassified,total_location\n")
        log_compiled.close()

        log_compiled = open(path + "evaluation/log_compiled_path.csv", "w")
        log_compiled.write("path_segment,recognized_after,times_misclassified,path_len\n")
        log_compiled.close()

        path_count = 0
        deviation = 0
        times_not_found_path = 0
        for row in log_recognized_path_segment.iterrows():
            if row[1]["recognized_after"] == 99999999:
                times_not_found_path = times_not_found_path + 1
            else:
                deviation = deviation + row[1]["recognized_after"]
            path_count = path_count + 1

        log_path_segment_misclassified["recognized_after"] = log_recognized_path_segment["recognized_after"]
        log_location_misclassified["times_misclassified_as"] = log_location_misclassification["times_misclassified_as"]

        # with portalocker.Lock(path + "evaluation/log_compiled.csv", "a",
        #                      timeout=60) as log_compiled:
        with open(path + "evaluation/log_compiled.csv", "a") as log_compiled:
            log_compiled.write(
                "{0},{1},{2},{3},{4},{5},{6},{7}\n".format(folder_prefix, log_true_vs_predicted.iloc[0]["accuracy"],
                                                           log_true_vs_predicted.iloc[0][
                                                               "accuracy_given_previous_location_was_correct"],
                                                           log_true_vs_predicted.iloc[0][
                                                               "accuracy_given_previous_location_was_incorrect"],
                                                           log_true_vs_predicted.iloc[0][
                                                               "accuracy_given_location_is_cont_the_same_and_within_5_entries"],
                                                           log_true_vs_predicted.iloc[0][
                                                               "accuracy_given_location_is_cont_the_same_and_within_10_entries"],
                                                           deviation / path_count, times_not_found_path))

        with open(path + "evaluation/log_compiled_location.csv", "a") as log_compiled:
            for row in log_location_misclassified.iterrows():
                log_compiled.write(
                    "{0},{1},{2},{3},{4}\n".format(folder_prefix, row[1]["location"], row[1]["times_misclassified_as"],
                                                   row[1]["times_misclassified"], row[1]["total_location"]))

        with open(path + "evaluation/log_compiled_path.csv", "a") as log_compiled:
            for row in log_path_segment_misclassified.iterrows():
                log_compiled.write(
                    "{0},{1},{2},{3},{4}\n".format(folder_prefix, row[1]["path_segment"], row[1]["recognized_after"],
                                                   row[1]["times_misclassified"], row[1]["path_len"]))
