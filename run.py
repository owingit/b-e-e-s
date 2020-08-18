import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections
import networkx as nx


class MasterPlot:
    def __init__(self, num_bees, side_length, thetastars, num_trials, identifier="variance"):
        self.thetastars = thetastars
        self.master_plot = collections.OrderedDict()
        self.identifier = identifier
        self.num_trials = num_trials
        self.y_label, self.title = self.set_labels(num_bees, side_length)

    def set_labels(self, num_bees, side_length):
        boilerplate = '(num_agents={}, side_length={}, num_trials={})'.format(
            num_bees, side_length, self.num_trials
        )
        plot_y_label = None
        plot_title = None
        if self.identifier == "variance":
            plot_y_label = "Variance in food level between individuals"
            plot_title = 'Variance in food level over time ' + boilerplate
        if self.identifier == "num_donation_events":
            plot_y_label = "Largest network of trophallaxis donors"
            plot_title = 'Maximum value of trophallaxis donor network over time ' + boilerplate
        if self.identifier == "stdev":
            plot_y_label = "Standard deviation in food level between individuals"
            plot_title = 'Standard deviation in food level over time ' + boilerplate
        if self.identifier == "max_donations":
            plot_y_label = "Max number of donations within an individual"
            plot_title = 'Amount of donations by biggest donor over time ' + boilerplate
        assert plot_y_label, "No y label set!"
        assert plot_title, "No title set!"
        return plot_y_label, plot_title

    def plot_timeseries_data(self):
        for thetastar in self.thetastars:
            for key, value in self.master_plot[thetastar].items():
                if type(value) is list:
                    new_val = sum(value) / len(value)
                    self.master_plot[thetastar][key] = new_val

            plt.plot(self.master_plot[thetastar].keys(), self.master_plot[thetastar].values(),
                     label='Thetastar: {}'.format(thetastar))
        plt.xlabel('Timestep')
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()
        plt.show()


def main():
    thetastars = [math.pi / 8, math.pi / 4]
    # , math.pi / 4, math.pi / 2]
    side_length = 100
    num_bees = 64
    step_count = 5000
    num_trials = 2

    variance_plot = MasterPlot(num_bees, side_length, thetastars, num_trials, "variance")
    max_donations_plot = MasterPlot(num_bees, side_length, thetastars, num_trials, "max_donations")
    cluster_size_plot = MasterPlot(num_bees, side_length, thetastars, num_trials, "num_donation_events")
    plots = [variance_plot, max_donations_plot, cluster_size_plot]

    for trial in range(0, num_trials):
        for thetastar in thetastars:
            simulation = Simulation.Simulation(num_bees=num_bees, side_length=side_length, step_count=step_count,
                                               thetastar=thetastar,
                                               initially_fed_percentage=0.25)
            simulation.run(food_donation_percent=0.50, food_transition_rate=1.0)
            simulation.animate_walk()
            largest_network = None
            max_edges = 0
            for bee in simulation.bee_array:
                if bee.trophallaxis_network:
                    if len(bee.trophallaxis_network.edges()) > max_edges:
                        max_edges = len(bee.trophallaxis_network.edges())
                        largest_network = bee.trophallaxis_network

            # if largest_network:
            #     pos = nx.spring_layout(largest_network)
            #     nx.draw(largest_network, pos)
            #     nx.draw_networkx_labels(largest_network, pos)
            #     # plt.show()
            for plot in plots:
                raw_stats = simulation.get_timeseries_stats(plot.identifier)
                if plot.master_plot.get(thetastar) is None:
                    plot.master_plot[thetastar] = raw_stats
                else:
                    instance_dict = plot.master_plot[thetastar]
                    for key in raw_stats.keys():
                        if instance_dict.get(key):
                            items = []
                            if type(plot.master_plot[thetastar][key]) is list:
                                items.extend(plot.master_plot[thetastar][key])
                            else:
                                items.append(plot.master_plot[thetastar][key])
                            items.append(raw_stats[key])
                            plot.master_plot[thetastar][key] = items
                        else:
                            plot.master_plot[thetastar][key] = raw_stats[key]

    for plot in plots:
        plot.plot_timeseries_data()


if __name__ == "__main__":
    main()
