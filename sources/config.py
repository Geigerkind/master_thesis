import sys

from sources.data.data_set import DataSet

BIN_FOLDER_PATH = "/home/shino/Uni/master_thesis/bin"
NUM_CORES = 8
WITHOUT_PREVIOUS_EDGE = False


def parse_cmd_args():
    _, encode_paths_between_as_location, dt_forest_size, dt_max_height, ffnn_num_hidden_layers, \
    ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs, input_data_sets, load_from_disk = sys.argv
    input_data_sets = input_data_sets.split(',')
    input_data_sets = [int(x) for x in input_data_sets]
    res_input_data_sets = []
    if 1 in input_data_sets:
        res_input_data_sets.append(DataSet.SimpleSquare)
    if 2 in input_data_sets:
        res_input_data_sets.append(DataSet.LongRectangle)
    if 3 in input_data_sets:
        res_input_data_sets.append(DataSet.RectangleWithRamp)
    if 4 in input_data_sets:
        res_input_data_sets.append(DataSet.ManyCorners)

    evaluation_name = "eval_{0}_DT_{1}_{2}_KNN_{3}_{4}_{5}_DS_{6}".format(encode_paths_between_as_location,
                                                                          dt_forest_size,
                                                                          dt_max_height, ffnn_num_hidden_layers,
                                                                          ffnn_num_nodes_per_hidden_layer,
                                                                          ffnn_num_epochs,
                                                                          "".join([str(x) for x in input_data_sets]))

    raw_encode_paths_between_as_location = encode_paths_between_as_location
    encode_paths_between_as_location = 1 == int(encode_paths_between_as_location)
    dt_forest_size = int(dt_forest_size)
    dt_max_height = int(dt_max_height)
    ffnn_num_hidden_layers = int(ffnn_num_hidden_layers)
    ffnn_num_nodes_per_hidden_layer = int(ffnn_num_nodes_per_hidden_layer)
    ffnn_num_epochs = int(ffnn_num_epochs)
    load_from_disk = int(load_from_disk) == 1
    pregen_path = BIN_FOLDER_PATH + "/pregen_data/data_" + raw_encode_paths_between_as_location + "_" + "".join(
        [str(x) for x in input_data_sets]) + ".pkl"

    return encode_paths_between_as_location, dt_forest_size, dt_max_height, ffnn_num_hidden_layers, ffnn_num_nodes_per_hidden_layer, ffnn_num_epochs, load_from_disk, pregen_path, evaluation_name, res_input_data_sets
