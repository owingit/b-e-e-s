import helpers
import random
import numpy as np
import networkx as nx


class Bee:
    #  Number, total number, theta*, array of x positions, array of y positions, box dimension, number of steps,
    #  starting distribution, whether initially fed
    def __init__(self, i, total, tstar, initially_fed,
                 x_positions, y_positions, n, steps, r_or_u):
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if r_or_u == "random":
            self.positionx[0] = random.randint(0, n)
            self.positiony[0] = random.randint(0, n)
        else:
            uniform_x_position, uniform_y_position = helpers.get_uniform_coordinates(i, n, total)
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position
        self.direction = np.zeros(steps)
        self.direction[0] = helpers.get_initial_direction(100)
        self.theta_star = tstar
        self.name = "Bee #{}".format(i)
        self.number = i

        if initially_fed:
            self.food_level = 100
        else:
            self.food_level = 0

        self.active_donor = None
        self.active_receiver = None
        self.steps_to_wait = 0

        self.placed = False
        self.bees_seen = {}  # dict mapping bee number: list of steps at which encountered
        self.food_in_edges = []  # list of bees that have fed this bee
        self.food_out_edges = []  # list of bees that have been fed by this bee
        self.agents_in_connected_component = [i]
        self.counts_for_encounter = True
