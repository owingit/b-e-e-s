import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
import numpy as np


side_length = int(raw_input("Side length?"))
filename = raw_input("Please provide a file and we'll plot the agent paths. ")
all_paths = np.load(filename)
num_agents = all_paths.shape[0]
num_steps = all_paths.shape[1]

dict_of_xs = {}
dict_of_ys = {}
dict_of_ccsizes = {}
colors_to_ccsize = {}

colors = dict(mcolors.XKCD_COLORS, **mcolors.CSS4_COLORS)
num_values = len(colors.keys())
x = np.arange(num_values)
zipped_colors = zip(x, colors.keys())


for agent in range(0, num_agents):
    dict_of_xs[agent] = []
    dict_of_ys[agent] = []
    dict_of_ccsizes[agent] = []
    colors_to_ccsize[agent] = []

for agent in range(0, num_agents):
    list_of_xs = []
    list_of_ys = []
    list_of_ccs = []
    for step in range(0, num_steps):
        list_of_xs.append(all_paths[agent][step][0])
        list_of_ys.append(all_paths[agent][step][1])
        list_of_ccs.append(all_paths[agent][step][2])
    dict_of_xs[agent].extend(list_of_xs)
    dict_of_ys[agent].extend(list_of_ys)
    dict_of_ccsizes[agent].extend(list_of_ccs)

print dict_of_ccsizes.values()

for key in dict_of_ccsizes.keys():
    for ccsize in dict_of_ccsizes[key]:
        colors_to_ccsize[key].append(str(zipped_colors[int(ccsize)][1]))

print colors_to_ccsize.values()


fig = plt.figure()
plt.xlim(0, side_length)
plt.ylim(0, side_length)
fig.suptitle('Paths over time colored by connected component size')


graphs = {}
for agent in range(0, num_agents):
    agent_graph, = plt.plot([], [], 'o')
    graphs[agent] = agent_graph


def animate(i):
    for k in graphs.keys():
        graphs[k].set_data(dict_of_xs[k][:i+1], dict_of_ys[k][:i+1])
        color = colors_to_ccsize[k][:i+1]
        graphs[k].set_color(colors_to_ccsize[k][:i+1][-1])
    return graphs.values()


ani = FuncAnimation(fig, animate, frames=num_steps, interval=500)
plt.show()