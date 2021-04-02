import matplotlib.pyplot as plt
import pandas as pd


def plot_route(name, file_name):
    route = pd.read_csv("/home/shino/Uni/master_thesis/bin/data/" + file_name)
    xyz = route[["x_pos", "y_pos", "z_pos"]]

    fig, ax = plt.subplots()
    ax.scatter(xyz["x_pos"], xyz["y_pos"], alpha=0.5)

    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_title('XY-Plot of {0}'.format(name))

    ax.grid(True)
    fig.tight_layout()

    plt.savefig("/home/shino/Uni/master_thesis/bin/route_{0}.png".format(name))


plot_route("simple_square", "simple_square.csv")
plot_route("long_rectangle", "long_rectangle.csv")
plot_route("rectangle_with_ramp", "rectangle_with_ramp.csv")
plot_route("many_corners", "many_corners.csv")
