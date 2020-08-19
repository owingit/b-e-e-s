import simulation_helpers
import numpy as np
import itertools
import Bee
import random
import matplotlib.pyplot as plt
import math
import statistics
from matplotlib.animation import FuncAnimation
import collections


class Simulation:
    def __init__(self, num_bees, side_length, step_count, thetastar, r_or_u="uniform",
                 use_periodic_boundary_conditions=True, initially_fed_percentage=0.1):
        self.total_bees = num_bees
        self.n = side_length
        self.steps = step_count
        self.bee_array = []
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        thetastars = [np.linspace(-thetastar, thetastar, simulation_helpers.TSTAR_RANGE)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.has_run = False

        # simulation parameters
        self.food_variance_threshold = 100
        self.min_food_level = 5.0
        self.r = 2.0

        self.food_variance_vs_time = collections.OrderedDict()
        self.max_donations_over_time = collections.OrderedDict()
        self.largest_cluster_over_time = collections.OrderedDict()
        self.cluster_count_over_time = collections.OrderedDict()

        for i in range(0, self.total_bees):
            self.bee_array.append(Bee.Bee(i, total=self.total_bees, tstar=self.thetastar,
                                          tstar_range=simulation_helpers.TSTAR_RANGE,
                                          initially_fed_percentage=initially_fed_percentage,
                                          n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                                          use_periodic_boundary_conditions=use_periodic_boundary_conditions))
        if all(bee.food_level == 0 for bee in self.bee_array):
            self.bee_array[0].set_food_level(100)
        self.init_stats()

    def init_stats(self):
        food_level_values = []
        donation_values = []
        for bee in self.bee_array:
            food_level_values.append(bee.food_level)
            donation_values.append(len(bee.food_out_edges))

        self.food_variance_vs_time[0] = statistics.variance(food_level_values)
        self.max_donations_over_time[0] = max(donation_values)
        self.largest_cluster_over_time[0] = 1

    def run(self, food_donation_percent=0.50, food_transition_rate=1):
        for step in range(1, self.steps):
            food_level_values = []
            donation_values = []
            cluster_sizes = []
            clusters = set()

            for bee in self.bee_array:
                bee.move(step)
                food_level_values.append(bee.food_level)
                donation_values.append(len(bee.food_out_edges))

            self.food_variance_vs_time[step] = statistics.variance(food_level_values)
            self.max_donations_over_time[step] = max(donation_values)
            for bee_1, bee_2 in itertools.combinations(self.bee_array, 2):
                dist = ((bee_1.positionx[step] - bee_2.positionx[step]) ** 2 +
                        (bee_1.positiony[step] - bee_2.positiony[step]) ** 2) ** 0.5
                simulation_helpers.populate_clusters(step, dist, bee_1, bee_2, self.r)
                simulation_helpers.adjust_direction_for_attraction(step, dist, bee_1, bee_2, self.r)
                simulation_helpers.setup_trophallaxis(step, dist, bee_1, bee_2,
                                                      food_donation_percent, food_transition_rate)

            if all([bee.food_level > self.min_food_level for bee in self.bee_array]):
                print("Convergence due to min food level reached after {} steps".format(step))
                break

            if self.food_variance_vs_time[step] <= self.food_variance_threshold:
                print("Convergence due to variance reached after {} steps".format(step))
                break

            for bee in self.bee_array:
                cluster_sizes.append(bee.cluster_size[step])
                clusters.add(frozenset(bee.cluster))
            self.cluster_count_over_time[step] = len(clusters)
            self.largest_cluster_over_time[step] = max(cluster_sizes)

        self.has_run = True

    def animate_walk(self):
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n), ylim=(0, self.n))
        xdatas = {n: [] for n in range(0, self.total_bees)}
        ydatas = {n: [] for n in range(0, self.total_bees)}

        bee_paths = [ax.plot([], [], '*')[0] for _ in self.bee_array]
        r_set = set(np.linspace(0, 1, num=self.total_bees))
        g_set = set(np.linspace(0, 1, num=self.total_bees))
        b_set = set(np.linspace(0, 1, num=self.total_bees))
        for line in bee_paths:
            r = random.sample(r_set, 1)[0]
            g = random.sample(g_set, 1)[0]
            b = random.sample(b_set, 1)[0]
            line.set_color((r, g, b))

        def animate(i, bees, lines):
            for line, bee in zip(lines, bees):
                xdatas[bee.number].append(bee.trace.get(i)[0])
                ydatas[bee.number].append(bee.trace.get(i)[1])
                line.set_data(xdatas[bee.number][0], ydatas[bee.number][0])
                xdatas[bee.number].pop(0)
                ydatas[bee.number].pop(0)
            return lines

        ax.set_xlim([0.0, self.n])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, self.n])
        ax.set_ylabel('Y')

        ax.set_title('2D Walk Test')

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=(self.bee_array, bee_paths),
                             interval=1000, blit=False)

        # anim.save('bee_paths.gif', writer='pillow')
        plt.show()

    def get_timeseries_stats(self, identifier="variance"):
        plot_dict = None
        secondary_plot_dict = None
        if identifier == "variance":
            plot_dict = self.food_variance_vs_time
        if identifier == "num_clusters":
            plot_dict = self.cluster_count_over_time
            secondary_plot_dict = self.largest_cluster_over_time
        if identifier == "stdev":
            plot_dict = self.food_variance_vs_time
            for key, value in plot_dict.items():
                plot_dict[key] = math.sqrt(value)
        if identifier == "max_donations":
            plot_dict = self.max_donations_over_time

        return plot_dict, secondary_plot_dict
