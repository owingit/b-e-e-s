import simulation_helpers
import random
import numpy as np
import math

import networkx as nx


class Bee:
    #  Number, total number, theta*, thetastar_range, box dimension, number of steps,
    #  starting distribution, whether initially fed, whether to use periodic boundary conditions
    def __init__(self, i, total, tstar, tstar_range,
                 initially_fed_percentage, n, steps, r_or_u, use_periodic_boundary_conditions):
        self.velocity = 1.0
        self.side_length_of_enclosure = n
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if r_or_u == "random":
            self.positionx[0] = random.randint(0, n)
            self.positiony[0] = random.randint(0, n)
        else:
            uniform_x_position, uniform_y_position = simulation_helpers.get_uniform_coordinates(i, n, total)
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position
        self.direction = np.zeros(steps)
        self.direction[0] = simulation_helpers.get_initial_direction(tstar_range)
        self.direction_set = False
        self.theta_star = tstar
        self.trace = {0: (self.positionx[0], self.positiony[0])}

        self.name = "Bee #{}".format(i)
        self.number = i
        if use_periodic_boundary_conditions:
            self.boundary_conditions = self.periodic_boundary_conditions
        else:
            self.boundary_conditions = self.non_periodic_boundary_conditions

        if random.random() < initially_fed_percentage:
            self.food_level = 100
        else:
            self.food_level = 0

        self.active_donor = False
        self.active_receiver = False
        self.is_engaged_in_trophallaxis = False
        self.food_to_give = 0
        self.food_to_receive = 0
        self.steps_to_wait = 0
        self.trophallaxis_network = nx.DiGraph()

        if self.food_level == 100:
            self.trophallaxis_network.add_node(self.number)

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
        elif self.direction_set:
            direction = self.direction[current_step - 1]
            self.direction_set = False
        else:
            direction = self.direction[current_step - 1] + step_theta

        if self.active_donor or self.active_receiver:
            self.engage()
            self.stay_put(current_step)
        else:
            self.attempt_step(current_step, direction)

    def engage(self):
        if self.food_to_give > 0:
            food_to_give_at_this_timestep = float(self.food_to_give / self.steps_to_wait)
            self.food_level = self.food_level - food_to_give_at_this_timestep
            self.food_to_give = self.food_to_give - food_to_give_at_this_timestep

        if self.food_to_receive > 0:
            food_to_receive_at_this_timestep = float(self.food_to_receive / self.steps_to_wait)
            self.food_level = self.food_level + food_to_receive_at_this_timestep
            self.food_to_receive = self.food_to_receive - food_to_receive_at_this_timestep

    def attempt_step(self, current_step, direction):
        self.direction[current_step] = direction
        potential_x_position = self.positionx[current_step - 1] + self.velocity * math.cos(direction)
        potential_y_position = self.positiony[current_step - 1] + self.velocity * math.sin(direction)
        self.complete_step(current_step, potential_x_position, potential_y_position)

    def complete_step(self, current_step, x, y):
        self.positionx[current_step] = x
        self.positiony[current_step] = y
        self.boundary_conditions(current_step)
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

    def stay_put(self, current_step):
        self.direction[current_step] = self.direction[current_step - 1]
        self.positionx[current_step] = self.positionx[current_step - 1]
        self.positiony[current_step] = self.positiony[current_step - 1]
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])
        self.steps_to_wait -= 1
        if self.steps_to_wait == 0:
            self.active_donor = None
            self.active_receiver = None
            self.is_engaged_in_trophallaxis = False

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

    def set_food_level(self, food_level):
        self.food_level = food_level
