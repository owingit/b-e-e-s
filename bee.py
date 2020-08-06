import helpers
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx


class Bee:
    #  Number, total number, theta*, box dimension, number of steps,
    #  starting distribution, whether initially fed, whether to use periodic boundary conditions
    def __init__(self, i, total, tstar, initially_fed, n, steps, r_or_u, use_periodic_boundary_conditions):
        self.velocity = 1.0
        self.side_length_of_enclosure = n
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
        self.trace = {0: (self.positionx, self.positiony)}

        self.name = "Bee #{}".format(i)
        self.number = i
        if use_periodic_boundary_conditions:
            self.boundary_conditions = self.periodic_boundary_conditions
        else:
            self.boundary_conditions = self.non_periodic_boundary_conditions

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

    def move(self, current_step):
        random_int = random.randint(0, 99)
        step_theta = self.theta_star[random_int]
        if current_step == 0:
            direction = self.direction[current_step]
        else:
            direction = self.direction[current_step - 1] + step_theta
        self.direction[current_step] = direction
        self.positionx[current_step] = self.positionx[current_step - 1] + self.velocity * math.cos(direction)
        self.positiony[current_step] = self.positiony[current_step - 1] + self.velocity * math.sin(direction)

        self.boundary_conditions(current_step)
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

    def stay_put(self, current_step):
        self.direction[current_step] = self.direction[current_step - 1]
        self.positionx[current_step] = self.positionx[current_step - 1]
        self.positiony[current_step] = self.positiony[current_step - 1]
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

    def periodic_boundary_conditions(self, current_step):
        if self.positionx[current_step] > self.side_length_of_enclosure:
            self.positionx[current_step] = self.positionx[current_step] - self.side_length_of_enclosure
        if self.positionx[current_step] < 0:
            self.positionx[current_step] += self.side_length_of_enclosure

        if self.positiony[current_step] > self.side_length_of_enclosure:
            self.positiony[current_step] = self.positiony[current_step] - self.side_length_of_enclosure
        if self.positiony[current_step] < 0:
            self.positiony[current_step] += self.side_length_of_enclosure

    def non_periodic_boundary_conditions(self, current_step):
        flip_direction = False
        if self.positionx[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positionx[current_step] - self.side_length_of_enclosure)
            self.positionx[current_step] = self.positionx[current_step] - 2 * distance_from_edge
            flip_direction = True
        if self.positionx[current_step] < 0:
            distance_from_edge = abs(0 - self.positionx[current_step])
            self.positionx[current_step] = self.positionx[current_step] + 2 * distance_from_edge
            flip_direction = True
        if self.positiony[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positiony[current_step] - self.side_length_of_enclosure)
            self.positiony[current_step] = self.positiony[current_step] - 2 * distance_from_edge
            self.direction[current_step] = -self.direction[current_step]
            flip_direction = True
        if self.positiony[current_step] < 0:
            distance_from_edge = abs(0 - self.positiony[current_step])
            self.positiony[current_step] = self.positiony[current_step] + 2 * distance_from_edge
            flip_direction = True

        if flip_direction:
            self.direction[current_step] = -self.direction[current_step]


def test_bees():
    bee_array = []
    total_bees = 100
    thetastar = [np.linspace(-(math.pi / 2), (math.pi / 2), 100)]
    thetastar = list(thetastar[0])
    n = 10
    steps = 10
    for i in range(0, total_bees):
        bee_array.append(Bee(i, total_bees, thetastar, False, n, steps, "uniform", True))
    for step in range(steps):
        for bee in bee_array:
            bee.move(step)
    for bee in bee_array:
        print(bee.trace.items())

test_bees()