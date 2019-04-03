import json
import matplotlib.pyplot as plt
import collections
import numpy as np
import math

FOOD_THRESHOLD = 5
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), 50) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]
counts = input("how many agents?")
side_length = input("how long were the sides?")
steps = input("how many steps?")
num_trials = input("how many trials?")


def read_and_plot_data():

    with open('steps_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp:
        steps_between_data = collections.OrderedDict(json.load(fp))
#    with open('encounters_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
#          counts, side_length, side_length, steps), 'r') as fp2:
#        encounters_data = collections.OrderedDict(json.load(fp2))
#    with open('unique_encounters_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
#          counts, side_length, side_length, steps), 'r') as fp3:
#        unique_encounters_data = collections.OrderedDict(json.load(fp3))
    with open('unique_encounters_up_to_stepcount_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp4:
        unique_encounters_up_to_stepcount_data = collections.OrderedDict(json.load(fp4))
    with open('gs_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp5:
        gs_up_to_stepcount_data = collections.OrderedDict(json.load(fp5))

    plot_avg_steps_between(steps_between_data, counts, side_length, steps, num_trials)
    #  plot_encounters(encounters_data, unique_encounters_data, counts, side_length, steps, num_trials)
    #  plot_encounters_up_to_stepcount(unique_encounters_up_to_stepcount_data, counts, side_length, steps, num_trials)
    plot_gs_up_to_stepcount_data(gs_up_to_stepcount_data, counts, side_length, num_trials)


def read_and_plot_food_data():

    with open('fed_bee_distribution_{}x{}_{}agents_TO_PLOT.json'.format(side_length, side_length, counts), 'r') as fp:
        food_data = collections.OrderedDict(json.load(fp))
    print food_data
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        food_data[thetastar] = {int(key): val for key, val in food_data[thetastar].items()}
        plt.loglog(sorted(food_data[thetastar].keys()), food_data[thetastar].values(), label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Number of fed individuals above threshold = {} units of food'.format(FOOD_THRESHOLD))
    plt.legend()
    plt.title('Number of fed individuals vs. time for {} agents in a {}x{} arena (n={})'.format(counts, side_length, side_length, num_trials))
    plt.show()
   

def plot_avg_steps_between(steps_between_data, counts, side_length, steps, num_trials):
    plt.plot(steps_between_data.keys(), steps_between_data.values())
    plt.xlabel('Thetastar')
    plt.ylabel('Avg steps between encounters')
    plt.title('Avg number of steps between encounters and between unique encounters for {} in {}x{} arena over {} steps (n={})'.format(
            counts,  side_length, side_length, steps, num_trials))
    plt.legend()
    plt.show()


def plot_encounters(encounters_data, unique_encounters_data, counts, side_length, steps, num_trials):
    fig, ax1 = plt.subplots()
    ax1.bar(encounters_data.keys(), encounters_data.values(), 0.2)
    ax2 = ax1.twinx()
    ax2.plot(unique_encounters_data.keys(), unique_encounters_data.values(), color='orange')
    ax1.set_xlabel('Theta* : the width of the range of possible angles')
    ax1.set_ylabel('Num of encounters (whole system)')
    ax2.set_ylabel('% unique encounters (whole system)')
    plt.title('Number of encounters between {} vs. theta* : {}x{} arena over {} steps (n={})'.format(
        counts, side_length, side_length, steps, num_trials))
    plt.show()


def plot_encounters_up_to_stepcount(unique_encounters_up_to_stepcount_data, counts, side_length, steps, num_trials):
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        unique_encounters_up_to_stepcount_data[thetastar] = {int(key): val for key, val in unique_encounters_up_to_stepcount_data[thetastar].items()}
        l = unique_encounters_up_to_stepcount_data[thetastar].values()
        print l
        percents = [x[0] * 100 for x in l]
        plt.plot(sorted(unique_encounters_up_to_stepcount_data[thetastar].keys()), percents, zorder=10,
                 label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('% Unique encounters')
    plt.twinx()
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        l = unique_encounters_up_to_stepcount_data[thetastar].values()
        totals = [x[1] for x in l]
        plt.scatter(sorted(unique_encounters_up_to_stepcount_data[thetastar].keys()), totals, alpha=0.5,
                    markersize=2, zorder=2, label='Thetastar: {}'.format(thetastar))
    plt.ylabel('Total encounters')
    plt.legend(loc=9)
    plt.title('Unique encounters vs. theta star for {} agents over time for {} steps in a {}x{} arena (n={})'.format(counts, steps, side_length, side_length, num_trials))
    plt.show()


def plot_gs_up_to_stepcount_data(gs_up_to_stepcount_data, counts, side_length, num_trials):
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        gs_up_to_stepcount_data[thetastar] = {int(key): val for key, val in gs_up_to_stepcount_data[thetastar].items()}
        plt.plot(sorted(gs_up_to_stepcount_data[thetastar].keys()),
                 gs_up_to_stepcount_data[thetastar].values(), label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Size of the largest connected component')
    plt.legend()
    plt.title(
        'Size of the largest connected component vs. time for {} agents in a {}x{} arena (n={})'.format(
            counts, side_length, side_length, num_trials))
    plt.show()


read_and_plot_data()
read_and_plot_food_data()
