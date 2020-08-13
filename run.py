import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections


class MasterPlot:
    def __init__(self, num_bees, side_length, thetastars, identifier="variance"):
        self.thetastars = thetastars
        self.master_plot = collections.OrderedDict()
        self.identifier = identifier
        self.y_label, self.title = self.set_labels(num_bees, side_length)

    def set_labels(self, num_bees, side_length):
        boilerplate = '(num_agents={}, side_length={})'.format(
            num_bees, side_length
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
        if self.identifier == "max_food_level":
            plot_y_label = "Maximum food level within an individual"
            plot_title = 'Max food level over time ' + boilerplate
        assert plot_y_label, "No y label set!"
        assert plot_title, "No title set!"
        return plot_y_label, plot_title

    def plot_timeseries_data(self):
        for thetastar in thetastars:
            plt.plot(self.master_plot[thetastar].keys(), self.master_plot[thetastar].values(), zorder=10,
                     label='Thetastar: {}'.format(thetastar))
        plt.xlabel('Timestep')
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()
        plt.show()

thetastars = [math.pi / 2, math.pi / 4, math.pi / 8]
side_length = 50
num_bees = 121
step_count = 1000

variance_plot = MasterPlot(num_bees, side_length, thetastars, "variance")
max_food_level_plot = MasterPlot(num_bees, side_length, thetastars, "max_food_level")
cluster_size_plot = MasterPlot(num_bees, side_length, thetastars, "num_donation_events")
plots = [variance_plot, max_food_level_plot, cluster_size_plot]

for thetastar in thetastars:
    simulation = Simulation.Simulation(num_bees=num_bees, side_length=side_length, step_count=step_count, thetastar=thetastar)
    simulation.run()
    for plot in plots:
        plot.master_plot[thetastar] = simulation.get_timeseries_stats(plot.identifier)

for plot in plots:
    plot.plot_timeseries_data()
