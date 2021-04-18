import matplotlib.pyplot as plt
import pandas as pd

from sources.config import BIN_FOLDER_PATH

"""
Tool to plot recorded paths, for visualization.
It just plots it on a xy plane. But this is sufficient to get an idea of the route and how the points are 
relative to each other in terms of actual distance.
I used this to put down the various sources for heat, sound, magnetic force etc.
"""


def plot_route(name, file_name):
    route = pd.read_csv(BIN_FOLDER_PATH + "/data/" + file_name)
    xyz = route[["x_pos", "y_pos", "z_pos"]]

    fig, ax = plt.subplots()
    ax.scatter(xyz["x_pos"], xyz["y_pos"], alpha=0.5)

    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_title('XY-Plot of {0}'.format(name))

    ax.grid(True)
    fig.tight_layout()

    plt.savefig(BIN_FOLDER_PATH + "/route_{0}.png".format(name))


plot_route("simple_square", "simple_square.csv")
plot_route("long_rectangle", "long_rectangle.csv")
plot_route("rectangle_with_ramp", "rectangle_with_ramp.csv")
plot_route("many_corners", "many_corners.csv")
