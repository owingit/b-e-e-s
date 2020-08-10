import numpy
import matplotlib.pyplot as plt
import math
import random


def get_uniform_coordinates(index, side_length, total):
    positionsx = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    positionsy = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    x, y = numpy.meshgrid(positionsx, positionsy)
    x_coords = x.flatten()
    y_coords = y.flatten()
    return x_coords[index], y_coords[index]


def test_initial_coordinates():
    total = 150
    side_length = 15
    for i in range(0, total):
        print(get_uniform_coordinates(i, side_length, total)[0], get_uniform_coordinates(i, side_length, total)[1])
        plt.scatter(get_uniform_coordinates(i, side_length, total)[0],
                    get_uniform_coordinates(i, side_length, total)[1])
    plt.show()


def get_initial_direction(theta_star_range):
    all_directions = numpy.linspace(-math.pi, math.pi, theta_star_range)
    return all_directions[random.randint(0, theta_star_range - 1)]


def setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate):
    dist = ((bee_1.positionx[step] - bee_2.positionx[step]) ** 2 + (
                bee_1.positiony[step] - bee_2.positiony[step]) ** 2) ** 0.5
    if bee_1.food_level > bee_2.food_level and \
            dist <= 1.0 and \
            not bee_1.is_engaged_in_trophallaxis and \
            not bee_2.is_engaged_in_trophallaxis:
        bee_1.is_engaged_in_trophallaxis = True
        bee_2.is_engaged_in_trophallaxis = True
        bee_2.active_receiver = True
        bee_1.active_donor = True
        bee_1.food_to_give = (bee_1.food_level - bee_2.food_level) * food_donation_percent
        bee_2.food_to_receive = bee_1.food_to_give
        bee_1.steps_to_wait = food_transition_rate * bee_1.food_to_give
        bee_2.steps_to_wait = bee_1.steps_to_wait

    if bee_1.food_level < bee_2.food_level and \
            dist <= 1.0 and \
            not bee_1.is_engaged_in_trophallaxis and \
            not bee_2.is_engaged_in_trophallaxis:
        bee_1.is_engaged_in_trophallaxis = True
        bee_2.is_engaged_in_trophallaxis = True
        bee_1.active_receiver = True
        bee_2.active_donor = True
        bee_2.food_to_give = (bee_2.food_level - bee_1.food_level) * food_donation_percent
        bee_1.food_to_receive = bee_2.food_to_give
        bee_2.steps_to_wait = food_transition_rate * bee_2.food_to_give
        bee_1.steps_to_wait = bee_2.steps_to_wait