import pandas as pd
import portalocker


class CompileLog:
    def __init__(self, path, folder_prefix):
        self.folder_prefix = folder_prefix

        path = path + folder_prefix
        log_true_vs_predicted = pd.read_csv(path + "/log_true_vs_predicted.csv")
        log_recognized_path_segment = pd.read_csv(path + "/log_recognized_path_segment.csv")
        log_path_segment_misclassified = pd.read_csv(path + "/log_path_segment_misclassified.csv")
        log_location_misclassified = pd.read_csv(path + "/log_location_misclassified.csv")
        log_location_misclassification = pd.read_csv(path + "/log_location_misclassification.csv")

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

        with portalocker.Lock("/home/shino/Uni/master_thesis/bin/evaluation/log_compiled.csv", "a",
                              timeout=60) as log_compiled:
            log_compiled.write(
                "{0},{1},{2},{3},{4},{5},{6}\n".format(folder_prefix, log_true_vs_predicted.iloc[0]["accuracy"],
                                                       log_true_vs_predicted.iloc[0][
                                                           "accuracy_given_previous_location_was_correct"],
                                                       log_true_vs_predicted.iloc[0][
                                                           "accuracy_given_location_is_cont_the_same_and_within_5_entries"],
                                                       log_true_vs_predicted.iloc[0][
                                                           "accuracy_given_location_is_cont_the_same_and_within_10_entries"],
                                                       deviation / path_count, times_not_found_path))

        with portalocker.Lock("/home/shino/Uni/master_thesis/bin/evaluation/log_compiled_location.csv", "a",
                              timeout=60) as log_compiled:
            for row in log_location_misclassified.iterrows():
                log_compiled.write(
                    "{0},{1},{2},{3},{4}\n".format(folder_prefix, row[1]["location"], row[1]["times_misclassified_as"],
                                                   row[1]["times_misclassified"], row[1]["total_location"]))

        with portalocker.Lock("/home/shino/Uni/master_thesis/bin/evaluation/log_compiled_path.csv", "a",
                              timeout=60) as log_compiled:
            for row in log_path_segment_misclassified.iterrows():
                log_compiled.write(
                    "{0},{1},{2},{3},{4}\n".format(folder_prefix, row[1]["path_segment"], row[1]["recognized_after"],
                                                   row[1]["times_misclassified"], row[1]["path_len"]))
