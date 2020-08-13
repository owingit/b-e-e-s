import simulation_helpers
import numpy as np
import itertools
import Bee
import random
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


class Simulation:
    def __init__(self, num_bees, side_length, step_count, thetastars, r_or_u="uniform",
                 use_periodic_boundary_conditions=True):
        self.total_bees = num_bees
        self.n = side_length
        self.steps = step_count
        self.bee_array = []
        self.r_or_u = r_or_u
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.has_run = False
        for i in range(0, self.total_bees):
            self.bee_array.append(Bee.Bee(i, total=self.total_bees, tstar=self.thetastar,
                                          tstar_range=simulation_helpers.TSTAR_RANGE,
                                          initially_fed_percentage=0.1, n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                                          use_periodic_boundary_conditions=use_periodic_boundary_conditions))

    def run(self, food_donation_percent=0.50, food_transition_rate=1):
        for step in range(1, self.steps):
            for bee in self.bee_array:
                bee.move(step)
            for bee_1, bee_2 in itertools.combinations(self.bee_array, 2):
                simulation_helpers.setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate)

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
