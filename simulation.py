import simulation_helpers
import itertools


def simulate(bee_array, step_count):
    food_donation_percent = 0.50
    food_transition_rate = 1

    for step in range(1, step_count):
        for bee in bee_array:
            bee.move(step)
        for bee_1, bee_2 in itertools.combinations(bee_array, 2):
            simulation_helpers.setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate)
