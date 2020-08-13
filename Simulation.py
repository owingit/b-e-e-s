import simulation_helpers
import itertools
import Bee
import random


class Simulation:
    def __init__(self, num_bees, side_length, step_count, thetastars, r_or_u="uniform",
                 use_periodic_boundary_conditions=True):
        self.total_bees = num_bees
        self.n = side_length
        self.steps = step_count
        self.bee_array = []
        self.r_or_u = r_or_u
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        for i in range(0, self.total_bees):
            self.bee_array.append(Bee.Bee(i, total=self.total_bees, tstar=self.thetastar,
                                          tstar_range=simulation_helpers.TSTAR_RANGE,
                                          initially_fed_percentage=0.1, n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                                          use_periodic_boundary_conditions=use_periodic_boundary_conditions))

    def simulate(self, food_donation_percent=0.50, food_transition_rate=1):
        for step in range(1, self.steps):
            for bee in self.bee_array:
                bee.move(step)
            for bee_1, bee_2 in itertools.combinations(self.bee_array, 2):
                simulation_helpers.setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate)
