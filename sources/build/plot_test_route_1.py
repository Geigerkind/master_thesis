import matplotlib.pyplot as plt
import pandas as pd

from sources.data.test_route_1 import get_test_route_1_labeled_by_xy

route = pd.read_csv("/home/shino/Uni/master_thesis/external_sources/trial_route_1_data/pos_data.txt")
xyz = route[["x_pos", "y_pos", "z_pos"]]

fig, ax = plt.subplots()
ax.scatter(xyz["x_pos"], xyz["y_pos"], alpha=0.5)

ax.set_xlabel('X', fontsize=15)
ax.set_ylabel('Y', fontsize=15)
ax.set_title('XY-Plot of Test Route 1')

ax.grid(True)
fig.tight_layout()

plt.savefig("/home/shino/Uni/master_thesis/bin/test_route_1.png")

# Now with color
data = get_test_route_1_labeled_by_xy(True, 0.2)

colors = {
    0: 'black',
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'red',
    5: 'purple',
    6: 'cyan',
    7: 'magenta',
    8: 'orange'
}

fig, ax = plt.subplots()
ax.scatter(xyz["x_pos"], xyz["y_pos"], c=data["label"].map(colors), alpha=0.5)

ax.set_xlabel('X', fontsize=15)
ax.set_ylabel('Y', fontsize=15)
ax.set_title('XY-Plot of Test Route 1 - Colored')

ax.grid(True)
fig.tight_layout()

plt.savefig("/home/shino/Uni/master_thesis/bin/test_route_1_colored.png")
