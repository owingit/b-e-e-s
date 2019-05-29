import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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
ccs = {}
existing_ccs = {}
for step in range(0, num_steps):
    existing_ccs[step] = []

for agent in range(0, num_agents):
    dict_of_xs[agent] = []
    dict_of_ys[agent] = []
    dict_of_ccsizes[agent] = []
    colors_to_ccsize[agent] = []
    ccs[agent] = []

for agent in range(0, num_agents):
    list_of_xs = []
    list_of_ys = []
    list_of_ccs = []
    list_of_actual_ccs = []
    for step in range(0, num_steps):
        list_of_xs.append(all_paths[agent][step][0])
        list_of_ys.append(all_paths[agent][step][1])
        list_of_ccs.append(all_paths[agent][step][2])
        list_of_actual_ccs.append(all_paths[agent][step][3])
        if len(all_paths[agent][step][3]) > 0:
            if all_paths[agent][step][3] not in existing_ccs[step]:
                existing_ccs[step].append(all_paths[agent][step][3])
    dict_of_xs[agent].extend(list_of_xs)
    dict_of_ys[agent].extend(list_of_ys)
    dict_of_ccsizes[agent].extend(list_of_ccs)
    ccs[agent].extend(list_of_actual_ccs)

print ccs

print existing_ccs
cc_locations = {}
for step in range(0, num_steps):
    cc_locations[step] = []
for step in existing_ccs.keys():
    if len(existing_ccs[step]) > 0:
        for cc in existing_ccs[step]:
            list_of_locations = []
            for agent in cc:
                list_of_locations.append((all_paths[agent][step][0], all_paths[agent][step][1]))
                if list_of_locations not in cc_locations[step]:
                    cc_locations[step].append(list_of_locations)

print cc_locations

items = [(key, max(value)) for (key, value) in dict_of_ccsizes.viewitems()]
values = [max(value) for value in dict_of_ccsizes.values()]

m = max(values)
print m

# # cols = dict(mcolors.XKCD_COLORS, **mcolors.CSS4_COLORS)
# # if cols: change from colors.colors to cols.values()
# colors = cm.get_cmap('plasma_r', m+1)
# num_values = len(colors.colors)
# x = np.arange(num_values)
# zipped_colors = zip(x, colors.colors)
#
# for key in dict_of_ccsizes.keys():
#     for ccsize in dict_of_ccsizes[key]:
#         colors_to_ccsize[key].append(zipped_colors[int(ccsize)])
#
# values_that_appear = []
# for i in range(0, num_steps):
#     for j in range(0, len(items)):
#         if (dict_of_ccsizes[j][i], colors_to_ccsize[j][i]) not in values_that_appear:
#             values_that_appear.append((dict_of_ccsizes[j][i], colors_to_ccsize[j][i]))
#
# values_that_appear.sort()
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# plt.xlim(0, side_length)
# plt.ylim(0, side_length)
# # fig.suptitle('Paths over time colored by connected component size')
#
# graphs = {}
# for agent in range(0, num_agents):
#     agent_graph, = plt.plot([], [], 'o')
#     graphs[agent] = agent_graph
#
#
# def animate(i):
#     for k in graphs.keys():
#         graphs[k].set_data(dict_of_xs[k][:i+1], dict_of_ys[k][:i+1])
#         graphs[k].set_color(colors_to_ccsize[k][:i+1][-1][1])
#     fig.suptitle('Paths over time colored by connected component size at step {}'.format(i))
#     return graphs.values()
#
#
# ani = FuncAnimation(fig, animate, frames=num_steps, interval=1000)
# ani.save('movie_{}.mp4'.format(filename))
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.2,
#                  box.width, box.height * 0.8])
# patches = []
# for i in range(0, len(values_that_appear)):
#     patch = mpatches.Patch(color=values_that_appear[i][1][1], label='{}'.format(values_that_appear[i][1][0]))
#     patches.append(patch)
# plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), ncol=10, loc='upper center')
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# plt.xlim(0, side_length)
# plt.ylim(0, side_length)
#
# cc_graphs = {}
# for step in range(0, num_steps):
#     cc_graphs[step] = []
#     for cc in range(0, len(existing_ccs[step])):
#         agent_graph, = plt.plot([], [], 'o')
#         cc_graphs[step].append(agent_graph)
#
#
# def animate_ccs(i):
#     for k in range(0, len(cc_locations[i])):
#         print i, cc_locations[i]
#         if len(cc_locations[i]) > 0:
#             print len(cc_locations[i])
#             cc_graphs[i][k].set_data(cc_locations[i][k][0], cc_locations[i][k][1])
#             cc_graphs[i][k].set_color(colors_to_ccsize[k][:i+1][-1][1])
#
#     return cc_graphs.values()
#
#
# anim = FuncAnimation(fig, animate_ccs, frames=num_steps, interval=1000)
#
# for i in range(0, len(values_that_appear)):
#     patch = mpatches.Patch(color=values_that_appear[i][1][1], label='{}'.format(values_that_appear[i][1][0]))
#     patches.append(patch)
# plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), ncol=10, loc='upper center')
# plt.show()

#  to do another thing, we need to create a new data structure that tracks the locations of all the connected components, regardless of size
#  dict with keys = number, values = locations of points within the cc
#  color them differently

#  regression to fit logistic growth