import json
import matplotlib.pyplot as plt
import collections
import numpy as np
import math

FOOD_THRESHOLD = 10
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), 50) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]
THETASTARS.extend([np.linspace(0, 0, 50)])
counts = input("how many agents?")
side_length = input("how long were the sides?")
steps = input("how many steps?")
num_trials = input("how many trials?")
food = raw_input("food? y/n?")


def get_convergence_time(thetastar):
    with open('Average_convergence_times_for_each_thetastar_{}x{}_{}agents.txt'.format(
              side_length, side_length, counts), 'r') as fp:
        convergence_times = collections.OrderedDict(json.load(fp))
    return convergence_times[thetastar]


def shrink_to_convergence(data, convergent_step):
    return {key: data[key] for key in list(data.keys())[:convergent_step]}


def read_and_plot_data():
    with open('steps_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp:
        steps_between_data = collections.OrderedDict(json.load(fp))
    with open('total_encounters_up_to_stepcount_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
              counts, side_length, side_length, steps), 'r') as fp2:
        encounters_data = collections.OrderedDict(json.load(fp2))
#    with open('unique_encounters_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
#          counts, side_length, side_length, steps), 'r') as fp3:
#        unique_encounters_data = collections.OrderedDict(json.load(fp3))
    with open('unique_steps_over_time_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
              counts, side_length, side_length, steps), 'r') as fp2:
        unique_steps_between_over_time_data = collections.OrderedDict(json.load(fp2))
    with open('steps_over_time_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp3:
        steps_between_over_time_data = collections.OrderedDict(json.load(fp3))
    with open('unique_encounters_up_to_stepcount_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp4:
        unique_encounters_up_to_stepcount_data = collections.OrderedDict(json.load(fp4))
    with open('gs_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
            counts, side_length, side_length, steps), 'r') as fp5:
        gs_up_to_stepcount_data = collections.OrderedDict(json.load(fp5))

    plot_avg_steps_between(steps_between_data, counts, side_length, steps, num_trials)
    #  plot_encounters(encounters_data, unique_encounters_data, counts, side_length, steps, num_trials)
    plot_encounters_up_to_stepcount(unique_encounters_up_to_stepcount_data, encounters_data, counts, side_length, steps, num_trials)
    plot_gs_up_to_stepcount_data(gs_up_to_stepcount_data, counts, side_length, num_trials)
    plot_steps_between_over_time(steps_between_over_time_data, counts, side_length, steps, num_trials)
    plot_unique_steps_between_over_time(unique_steps_between_over_time_data, counts, side_length, steps, num_trials)


def read_and_plot_food_data():

    with open('fed_bee_distribution_{}x{}_{}agents_TO_PLOT.json'.format(side_length, side_length, counts), 'r') as fp:
        food_data = collections.OrderedDict(json.load(fp))
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        food_data[thetastar] = {int(key): val for key, val in food_data[thetastar].items()}
        convergence_time = get_convergence_time(thetastar)
        first_x_pairs = shrink_to_convergence(food_data[thetastar], convergence_time)
        plt.loglog(sorted(first_x_pairs.keys()), first_x_pairs.values(), label='Thetastar: {}'.format(thetastar))
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


def plot_encounters_up_to_stepcount(unique_encounters_up_to_stepcount_data, total_data, counts, side_length, steps, num_trials):
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        unique_encounters_up_to_stepcount_data[thetastar] = {int(key): val for key, val in unique_encounters_up_to_stepcount_data[thetastar].items()}
        total_data[thetastar] = {int(key): val for key, val in total_data[thetastar].items()}
        # convergence_time = get_convergence_time(thetastar)
        # first_x_pairs = shrink_to_convergence(unique_encounters_up_to_stepcount_data[thetastar], convergence_time)
        # plt.plot(sorted(first_x_pairs.keys()), first_x_pairs.values(), zorder=10,
        #          label='Thetastar: {}'.format(thetastar))
        plt.scatter(sorted(total_data[thetastar].keys()), total_data[thetastar].values(), zorder=0,
                 label='Thetastar: {}'.format(thetastar), marker='x')
        plt.plot(sorted(unique_encounters_up_to_stepcount_data[thetastar].keys()), unique_encounters_up_to_stepcount_data[thetastar].values(), zorder=10,
                 label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Unique encounters')
    plt.title('Total unique encounters vs. theta star for {} agents over time for {} steps in a {}x{} arena (n={})'.format(counts, steps, side_length, side_length, num_trials))
    plt.legend()
    plt.show()


def plot_steps_between_over_time(total, counts, side_length, steps, num_trials):
    print "Reg steps: ", total
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        total[thetastar] = {int(key): val for key, val in total[thetastar].items()}
        # convergence_time = get_convergence_time(thetastar)
        # first_x_pairs = shrink_to_convergence(total[thetastar], convergence_time)
        # plt.plot(sorted(first_x_pairs.keys()), first_x_pairs.values(), zorder=10,
        #          label='Thetastar: {}'.format(thetastar))
        plt.plot(sorted(total[thetastar].keys()), total[thetastar].values(), zorder=10,
                 label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Avg steps between encounters')
    plt.title('Steps between encounters over time for each thetastar: {} agents, {} steps in a {}x{} arena (n={})'.format(
        counts, steps, side_length, side_length, num_trials))
    plt.legend()
    plt.show()


def plot_unique_steps_between_over_time(total, counts, side_length, steps, num_trials):
    print "Unique steps: ", total
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        total[thetastar] = {int(key): val for key, val in total[thetastar].items()}
        # convergence_time = get_convergence_time(thetastar)
        # first_x_pairs = shrink_to_convergence(total[thetastar], convergence_time)
        plt.loglog(sorted(total[thetastar].keys()), total[thetastar].values(), zorder=10,
                   label='Thetastar: {}'.format(thetastar))
        # plt.hist(sorted(total[thetastar].values()), bins=50, histtype='step')
    plt.xlabel('# of steps')
    plt.ylabel('Freq')
    plt.title('Steps between unique encounters over time for each thetastar: {} agents, {} steps in a {}x{} arena (n={})'.format(
            counts, steps, side_length, side_length, num_trials))
    plt.legend()
    plt.show()


def plot_gs_up_to_stepcount_data(gs_up_to_stepcount_data, counts, side_length, num_trials):
    # twin axis with average food level
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        gs_up_to_stepcount_data[thetastar] = {int(key): val for key, val in gs_up_to_stepcount_data[thetastar].items()}
        # convergence_time = get_convergence_time(thetastar)
        # first_x_pairs = shrink_to_convergence(gs_up_to_stepcount_data[thetastar], convergence_time)
        # plt.plot(sorted(first_x_pairs.keys()),
        #            first_x_pairs.values(), label='Thetastar: {}'.format(thetastar))
        plt.plot(sorted(gs_up_to_stepcount_data[thetastar].keys()), gs_up_to_stepcount_data[thetastar].values(),
                 label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Size of the largest connected component')
    plt.legend()
    plt.title(
        'Size of the largest connected component vs. time for {} agents in a {}x{} arena (n={})'.format(
            counts, side_length, side_length, num_trials))
    plt.show()


read_and_plot_data()
if food == 'y':
    read_and_plot_food_data()
