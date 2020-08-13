import numpy
import matplotlib.pyplot as plt
import math
import random


TSTAR_RANGE = 100


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


def donor_logic(donor_bee, receiver_bee, food_donation_percent, food_transition_rate):
    donor_bee.is_engaged_in_trophallaxis = True
    receiver_bee.is_engaged_in_trophallaxis = True

    donor_bee.active_donor = True
    receiver_bee.active_receiver = True

    donor_bee.food_out_edges.append(receiver_bee.number)
    receiver_bee.food_in_edges.append(donor_bee.number)

    donor_bee.food_to_give = (donor_bee.food_level - receiver_bee.food_level) * food_donation_percent
    receiver_bee.food_to_receive = donor_bee.food_to_give
    donor_bee.steps_to_wait = food_transition_rate * donor_bee.food_to_give
    receiver_bee.steps_to_wait = donor_bee.steps_to_wait
    
    donor_bee.agents_in_connected_component.append(receiver_bee.number)
    receiver_bee.agents_in_connected_component = donor_bee.agents_in_connected_component


def setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate):
    dist = ((bee_1.positionx[step] - bee_2.positionx[step]) ** 2 + (
                bee_1.positiony[step] - bee_2.positiony[step]) ** 2) ** 0.5
    if bee_1.food_level > bee_2.food_level and \
            dist <= 1.0 and \
            not bee_1.is_engaged_in_trophallaxis and \
            not bee_2.is_engaged_in_trophallaxis:
        donor_logic(bee_1, bee_2, food_donation_percent, food_transition_rate)

    if bee_1.food_level < bee_2.food_level and \
            dist <= 1.0 and \
            not bee_1.is_engaged_in_trophallaxis and \
            not bee_2.is_engaged_in_trophallaxis:
        donor_logic(bee_2, bee_1, food_donation_percent, food_transition_rate)
