from webweb import Web
import numpy as np
import os, fnmatch
import collections
import math
import matplotlib.pyplot as plt
import networkx as nx

THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), 50) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]
network_steps = [0, 10, 20, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 2000, 3000]


side_length = input("Which side length?")
steps = input("How many agents?")

web = Web(title='all_graphs_{}_{}x{}'.format(steps, side_length, side_length))

degree_distribution_dict = collections.OrderedDict()
for thetastar in THETASTARS:
    ts = thetastar[-1] - thetastar[0]
    degree_distribution_dict[ts] = []
for thetastar in THETASTARS:
    ts = thetastar[-1] - thetastar[0]
    for network_step in network_steps[1:]:
        try:
            g = (nx.read_gml('networks/networks_thetastar_{}_{}x{}_{}steps/network_at_step_{}.gml'.format(
                      ts, side_length, side_length, steps, network_step)))
        except IOError:
            continue

        node_list = g.nodes()
        node_list = [int(node) for node in node_list]
        edge_list = g.edges(data='weight')
        edge_list = [(int(edge[0]), int(edge[1]), float(edge[2])) for edge in edge_list]
        if ts == THETASTARS[7][-1] - THETASTARS[7][0]:
            web.networks.pi_over_6.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[6][-1] - THETASTARS[6][0]:
            web.networks.pi_over_4.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[5][-1] - THETASTARS[5][0]:
            web.networks.pi_over_3.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[4][-1] - THETASTARS[4][0]:
            web.networks.pi_over_2.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[3][-1] - THETASTARS[3][0]:
            web.networks.two_pi_over_three.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[2][-1] - THETASTARS[2][0]:
            web.networks.pi.add_layer(adjacency=edge_list, nodes=node_list)
        elif ts == THETASTARS[1][-1] - THETASTARS[1][0]:
            web.networks.three_pi_over_two.add_layer(adjacency=edge_list, nodes=node_list)
        else:
            web.networks.two_pi.add_layer(adjacency=edge_list, nodes=node_list)
        degree_distribution_dict[ts].append((g, 'step {}'.format(network_step)))
web.display.colorBy = 'degree'
web.display.sizeBy = 'degree'
web.display.gravity = 0.5
web.show()

pattern = "network_convergence_*.gml"

for thetastar in THETASTARS:
    gs_at_convergence = []
    ts = thetastar[-1] - thetastar[0]
    listOfFiles = os.listdir('networks/networks_thetastar_{}_{}x{}_{}steps'.format(ts, side_length, side_length, steps))
    for file in listOfFiles:
        if fnmatch.fnmatch(file, pattern):
            g = nx.read_gml('networks/networks_thetastar_{}_{}x{}_{}steps/{}'.format(ts, side_length, side_length, steps, file))
            gs_at_convergence.append((g, convergence)

    degree_distribution_dict[ts].extend(gs_at_convergence)

for ts in degree_distribution_dict.keys():
    for (g, step) in degree_distribution_dict[ts]:
#        plt.xscale('log')
        plt.yscale('log')
        gdc = nx.degree_centrality(g)
        gd = nx.degree(g)
        gdc_plot = dict(collections.Counter(gdc.values()))
        gdc_plot = {float(key)+(1/steps): value for key, value in gdc_plot.items()}
        plt.scatter(list(gdc_plot.keys()), list(gdc_plot.values()), c='b',marker='x')
        plt.xlabel('Degree')
        plt.ylabel('Number of Agents')
        plt.title('Degree distribution at {} for a trial @ theta* {}'.format(step, ts))
        plt.show()
