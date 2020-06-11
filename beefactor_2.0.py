from __future__ import division
import matplotlib.pyplot as plt
import collections
import itertools
import networkx as nx

import sys
import threading
import time

import random
import numpy as np
import math

import json
import argparse

###########################################################
#           CONSTANTS AND GLOBAL DATA STRUCTURES          #
###########################################################


def near(limit):
    val = int(math.pow(limit, .5))
    return val*val


NETWORK_STRING = "{}/network_at_step_{}.gml"

GS = []
unique_encounters_up_to_stepcount = collections.OrderedDict()
total_encounters_up_to_stepcount = collections.OrderedDict()
gs_up_to_stepcount = collections.OrderedDict()
steps_to_encounter = collections.OrderedDict()
steps_to_encounter_over_time = collections.OrderedDict()
steps_to_unique_encounter_over_time = collections.OrderedDict()
food_distribution_vs_time = collections.OrderedDict()
networks_per_thread = collections.OrderedDict()
convergence_times = collections.OrderedDict()
unique_encounters_per_thread = collections.OrderedDict()

# constants from user input
COUNTS = int(input("How many agents? (Enter a perfect square, please :P) "))
STEPS = int(input("How many steps?"))
step_counts = [s for s in range(0, STEPS)]
network_steps = np.arange(0, STEPS, 10)
side_length = int(input("How long is each side of the arena?"))

NS = [25, 50, 75, 100, 150, 200, 250, 500]
NUM_TRIALS = 5
vel = 1.0  # step size, or velocity
PCT_INITIALLY_CARRYING_FOOD = 10
FOOD_TRANSFER_RATE = 1  # units / timestep
FOOD_THRESHOLD = 10
VARIANCE_THRESHOLD = 0.8
# sl2 = (side_length * side_length) / 2
NUM_CELLS = near(side_length * 2)
NUM_CELLS_PER_ROW = math.sqrt(NUM_CELLS)

THETASTARRANGE = 100
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), THETASTARRANGE) for i in
              (1, 1.5, 2, 6, 8, 10, 12, 16)]
THETASTARS.extend([np.linspace(0, 0, THETASTARRANGE)])
INITIAL_DIRECTION = np.linspace(-math.pi, math.pi, THETASTARRANGE)

lock = threading.RLock()


###########################################################
#                       NOTES                             #
###########################################################


# stopping conditions:
# a) global variance in food level should be less than some epsilon
# b) individual variance in food level should be less than some epsilon

# start making the networks again

# full bees should start near each other sorta done
# do they change orientation upon encounters? and if so, do they change back? or continue from the new trajectory?
# add food sources of various qualities that influence food transfer time / rate
# donor vs. expecter, where donors are stationary and expecters are mobile
# change the boundary conditions to allow for non-periodic boundary conditions


###########################################################
#                   CLASS DEFINITIONS                     #
###########################################################


class BeeThread(threading.Thread):
    def __init__(self, activate_food, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.is_tracking_food = activate_food
        self.thread_id = thread_id
        self.name = name
        self.counter = counter

    def run(self):
        print "Starting trial {}\n".format(self.thread_id)
        run_everything(self.is_tracking_food, self.name)


# Bee object
class Bee:
    def __init__(self, i, tstar, x_positions, y_positions, n, steps, r_or_u, initially_fed):
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if r_or_u == 1:
            if initially_fed:
                # figure this out better
                if (2 * PCT_INITIALLY_CARRYING_FOOD) > n:
                    self.positionx[0] = random.randint(0, n)
                    self.positiony[0] = random.randint(0, n)
                else:
                    self.positionx[0] = random.randint(n - 2 * PCT_INITIALLY_CARRYING_FOOD,
                                                       n - PCT_INITIALLY_CARRYING_FOOD)
                    self.positiony[0] = random.randint(n - 2 * PCT_INITIALLY_CARRYING_FOOD,
                                                       n - PCT_INITIALLY_CARRYING_FOOD)
            else:
                self.positionx[0] = random.randint(0, n)
                self.positiony[0] = random.randint(0, n)
        else:
            self.positionx[0] = x_positions[i]
            self.positiony[0] = y_positions[i]
        self.direction = np.zeros(steps)
        self.direction[0] = INITIAL_DIRECTION[random.randint(0, THETASTARRANGE - 1)]
        self.thetastar = tstar
        self.name = "Bee #{}".format(i)
        self.number = i
        if initially_fed:
            self.food_level = 100
        else:
            self.food_level = 0
        self.donor = None
        self.receiver = None
        self.steps_to_wait = 0
        self.placed = False
        self.step_count_of_last_encounter = [0]
        self.step_count_of_last_unique_encounter = [0]
        self.agents_in_connected_component = [i]
        self.counts_for_encounter = True


###########################################################
#                   HELPER FUNCTIONS                      #
###########################################################


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line

    :param mypath:
    """

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def find_adjacent(x, y, m, n):
    """
    Given an MxN matrix stored in a 1-d sequence,
    return the index of (possibly wrapped) X,Y
    """
    return (y % n) * m + (x % m)


def check_x_boundary(coords, n):
    a = False
    b = False
    if coords[0] > n - 1:
        a = True
    if coords[0] < 1:
        b = True
    return a, b


def get_thetastar(thetastar_range):
    return thetastar_range[-1] - thetastar_range[0]


def check_y_boundary(coords, n):
    a = False
    b = False
    if coords[1] > n - 1:
        a = True
    if coords[1] < 1:
        b = True
    return a, b


def get_adjacent_cells(x, y):
    adjacentList = []
    for dx in (-1, 0, +1):
        for dy in (-1, 0, +1):
            if dx != 0 or dy != 0:
                adjacentList.append(find_adjacent(x + dx, y + dy, 5, 5))
    return adjacentList


def initialize_result_dictionaries():
    for ts in THETASTARS:
        thetastar = get_thetastar(ts)
        steps_to_encounter_over_time[thetastar] = collections.OrderedDict()
        steps_to_unique_encounter_over_time[thetastar] = collections.OrderedDict()
        unique_encounters_up_to_stepcount[thetastar] = collections.OrderedDict()
        total_encounters_up_to_stepcount[thetastar] = collections.OrderedDict()
        gs_up_to_stepcount[thetastar] = collections.OrderedDict()


############################################################
#                   DATA PROCESSING                        #
############################################################


def write_data(is_tracking_food, avg_convergence_times):
    json_msg = '_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(COUNTS, side_length, side_length,
                                                                     STEPS)
    with open('steps' + json_msg, 'w') as fp:
        json.dump(steps_to_encounter.items(), fp, sort_keys=True)
    with open('unique_encounters_up_to_stepcount' + json_msg, 'w') as fp4:
        json.dump(unique_encounters_up_to_stepcount.items(), fp4, sort_keys=True)
    with open('total_encounters_up_to_stepcount' + json_msg, 'w') as fp5:
        json.dump(total_encounters_up_to_stepcount.items(), fp5, sort_keys=True)
    with open('gs' + json_msg, 'w') as fp6:
        json.dump(gs_up_to_stepcount.items(), fp6, sort_keys=True)
    with open('steps_over_time' + json_msg, 'w') as fp666:
        json.dump(steps_to_encounter_over_time.items(), fp666, sort_keys=True)
    with open('unique_steps_over_time'+json_msg, 'w') as fp777:
        json.dump(steps_to_unique_encounter_over_time.items(), fp777, sort_keys=True)
    if is_tracking_food:
        with open('fed_bee_distribution_{}x{}_{}agents_TO_PLOT.json'.format(side_length, side_length,
                                                                            COUNTS), 'w') as fp:
            json.dump(food_distribution_vs_time.items(), fp, sort_keys=True)
    with open('Average_convergence_times_for_each_thetastar_{}x{}_{}agents.txt'.format(side_length,
                                                                                       side_length,
                                                                                       COUNTS),
              'w') as fpwhatever:
        json.dump(avg_convergence_times.items(), fpwhatever, sort_keys=True)


def setup_results(is_tracking_food):
    for thetastar in THETASTARS:

        ts = get_thetastar(thetastar)
        for step in gs_up_to_stepcount[ts].keys():
            gs_up_to_stepcount[ts][step] = gs_up_to_stepcount[ts][step] / (
                        NUM_TRIALS)

        for sc in total_encounters_up_to_stepcount[ts].keys():
            if sc in total_encounters_up_to_stepcount[ts]:
                total_encounters_up_to_stepcount[ts][sc] = total_encounters_up_to_stepcount[ts][sc] // (
                            NUM_TRIALS)

            if sc in unique_encounters_up_to_stepcount[ts]:
                unique_encounters_up_to_stepcount[ts][sc] = unique_encounters_up_to_stepcount[ts][
                                                                sc] // (
                                                                        NUM_TRIALS)

            if sc in steps_to_encounter_over_time[ts]:
                steps_to_encounter_over_time[ts][sc] = steps_to_encounter_over_time[ts][sc] // (NUM_TRIALS)
            if sc in steps_to_unique_encounter_over_time[ts]:
                steps_to_unique_encounter_over_time[ts][sc] = steps_to_unique_encounter_over_time[ts][sc] // (NUM_TRIALS)

    for key in steps_to_encounter.keys():
        steps_to_encounter[key][0] = steps_to_encounter[key][0] / NUM_TRIALS
        steps_to_encounter[key][1] = steps_to_encounter[key][1] / NUM_TRIALS

    if is_tracking_food:
        for ths in food_distribution_vs_time.keys():
            food_converged_at_this_ts = 0
            for step in food_distribution_vs_time[ths].keys():
                food_distribution_vs_time[ths][step] = food_distribution_vs_time[ths][step] / (
                            NUM_TRIALS - food_converged_at_this_ts)
                # for key in convergence_times.keys():
                #     if step == convergence_times[key][ths]:
                #         food_converged_at_this_ts += 1

    avgs_at_thetastar = {}
    conv_at_thread = {}
    for thread_name in convergence_times.keys():
        for ths in convergence_times[thread_name].keys():
            if ths in avgs_at_thetastar:
                avgs_at_thetastar[ths] += convergence_times[thread_name][ths]
            else:
                avgs_at_thetastar[ths] = convergence_times[thread_name][ths]
            conv_at_thread[ths] = {thread_name: convergence_times[thread_name][ths]}

    for thetastar in avgs_at_thetastar.keys():
        avgs_at_thetastar[thetastar] = avgs_at_thetastar[thetastar] // NUM_TRIALS
    return avgs_at_thetastar


###########################################################
#                   ENCOUNTER LOGIC                       #
###########################################################


def feed(bee_array):
    engaged_bees = [bee for bee in bee_array if bee.donor or bee.receiver is not None]
    seen_donors = []
    for bee in engaged_bees:
        donor = bee.donor
        if donor in seen_donors:
            continue
        receiver = bee.receiver
        if bee_array[donor].food_level <= bee_array[receiver].food_level:
            continue
        else:
            bee_array[donor].food_level -= FOOD_TRANSFER_RATE
            bee_array[receiver].food_level += FOOD_TRANSFER_RATE
            seen_donors.append(donor)


def what_happens_during_an_encounter(g, is_tracking_food, current_step, bee_array, a, b,
                                     unique_encounters=None, encounters_at_this_step=None):
    # encounter logic
    if a not in encounters_at_this_step and b not in encounters_at_this_step:
        if bee_array[a].counts_for_encounter and bee_array[b].counts_for_encounter:
            encounters_at_this_step.extend([a, b])
            bee_array[a].step_count_of_last_encounter.append(current_step)
            bee_array[b].step_count_of_last_encounter.append(current_step)
            if not is_tracking_food:
                if g.has_edge(a, b) or g.has_edge(b, a):
                    g.edges[a, b]['weight'] += 1
                else:
                    g.add_edge(a, b, weight=1)

            # trophallaxis logic
            else:
                if bee_array[a].food_level > bee_array[b].food_level:
                    if bee_array[a].donor is None and bee_array[b].receiver is None:
                    # trophallaxis encounter
    	            # expand the networks ONLY DURING TROPHALLAXIS
                        if g.has_edge(a, b) or g.has_edge(b, a):
                            g.edges[a, b]['weight'] += 1
                        else:
                            g.add_edge(a, b, weight=1)
                            bee_array[a].agents_in_connected_component.extend(
                                bee_array[b].agents_in_connected_component)
                            bee_array[b].agents_in_connected_component = bee_array[a].agents_in_connected_component

                        bee_array[a].donor = a
                        bee_array[a].receiver = b
                        bee_array[b].donor = a
                        bee_array[b].receiver = b
                        steps_while_feeding = int((bee_array[a].food_level - bee_array[
                            b].food_level) / 2) / FOOD_TRANSFER_RATE
                        bee_array[a].steps_to_wait = steps_while_feeding
                        bee_array[b].steps_to_wait = steps_while_feeding
                        bee_array[a].counts_for_encounter = False
                        bee_array[b].counts_for_encounter = False
                elif bee_array[a].food_level < bee_array[b].food_level:
                    if bee_array[a].donor is None and bee_array[b].receiver is None:
                    # trophallaxis encounter
    	            # expand the networks ONLY DURING TROPHALLAXIS
                        if g.has_edge(a, b) or g.has_edge(b, a):
                            g.edges[a, b]['weight'] += 1
                        else:
                            g.add_edge(a, b, weight=1)
                            bee_array[a].agents_in_connected_component.extend(bee_array[b].agents_in_connected_component)
                            bee_array[b].agents_in_connected_component = bee_array[a].agents_in_connected_component

                        bee_array[a].donor = b
                        bee_array[a].receiver = a
                        bee_array[b].donor = b
                        bee_array[b].receiver = a
                        steps_while_feeding = int((bee_array[b].food_level - bee_array[
                            a].food_level) / 2) / FOOD_TRANSFER_RATE
                        bee_array[a].steps_to_wait = steps_while_feeding
                        bee_array[b].steps_to_wait = steps_while_feeding
                        bee_array[a].counts_for_encounter = False
                        bee_array[b].counts_for_encounter = False
                else:
                    bee_array[a].donor = None
                    bee_array[a].receiver = None
                    bee_array[b].donor = None
                    bee_array[b].receiver = None
                    bee_array[a].steps_to_wait = 0
                    bee_array[b].steps_to_wait = 0
                    bee_array[a].counts_for_encounter = True
                    bee_array[b].counts_for_encounter = True

            if (a, b) not in unique_encounters or (b, a) not in unique_encounters:
                unique_encounters.append((a, b))
                bee_array[a].step_count_of_last_unique_encounter.append(current_step)
                bee_array[b].step_count_of_last_unique_encounter.append(current_step)


def engage(is_tracking_food, all_paths, g, current_step, n, bee_array, cells, thread_name, running_total,
           unique_encounters):
    encounters_at_this_step = []
    exclude_cells = {}
    for cell in cells:
        locations_to_check = {}
        neighbor_bees_to_cell = []
        bees_in_cell = cells[cell]['occupants']
        if len(bees_in_cell) == 0:
            exclude_cells[cell] = True
            continue
        for index in cells[cell]['neighbor_indices']:
            if exclude_cells.get(index) is None:
                neighbor_bees_to_cell.extend(cells[index]['occupants'])
        for bee in bees_in_cell:
            locations_to_check[bee] = tuple(all_paths[bee][current_step])[:2]
        for bee in neighbor_bees_to_cell:
            locations_to_check[bee] = tuple(all_paths[bee][current_step])[:2]
        for (a, ordered_pair_a), (b, ordered_pair_b) in itertools.combinations(
                locations_to_check.items(), 2):

            # a, b are bee numbers
            modified = False

            # correct for periodic boundary conditions, then return values back to what they were
            shrink_ax, grow_ax = check_x_boundary(ordered_pair_a, n)
            shrink_bx, grow_bx = check_x_boundary(ordered_pair_b, n)

            shrink_ay, grow_ay = check_y_boundary(ordered_pair_a, n)
            shrink_by, grow_by = check_y_boundary(ordered_pair_b, n)

            if any([shrink_ax, shrink_bx, grow_ax, grow_bx, shrink_ay, grow_ay, shrink_by, grow_by]):
                modified = True

            op_a = list(ordered_pair_a)
            op_b = list(ordered_pair_b)

            if shrink_ax:
                op_a[0] = op_a[0] - (n - 1)
            if grow_ax:
                op_a[0] = op_a[0] + (n - 1)
            if shrink_ay:
                op_a[1] = op_a[1] - (n - 1)
            if grow_ay:
                op_a[1] = op_a[1] + (n - 1)

            if shrink_bx:
                op_b[0] = op_b[0] - (n - 1)
            if grow_bx:
                op_b[0] = op_b[0] + (n - 1)
            if shrink_by:
                op_b[1] = op_b[1] - (n - 1)
            if grow_by:
                op_b[1] = op_b[1] + (n - 1)

            if math.hypot(op_b[0] - op_a[0], op_b[1] - op_a[1]) <= vel and modified:
                what_happens_during_an_encounter(g, is_tracking_food, current_step, bee_array, a, b,
                                                 unique_encounters, encounters_at_this_step)

            if math.hypot(ordered_pair_b[0] - ordered_pair_a[0],
                          ordered_pair_b[1] - ordered_pair_a[1]) <= vel and not modified:
                what_happens_during_an_encounter(g, is_tracking_food, current_step, bee_array, a, b,
                                                 unique_encounters, encounters_at_this_step)
        exclude_cells[cell] = True

    lock.acquire()

    # add data to dicts for plotting later
    if len(g.edges()) > 0:
        thetastar_to_save = get_thetastar(bee_array[0].thetastar)
        if current_step in gs_up_to_stepcount[thetastar_to_save]:
            gs_up_to_stepcount[thetastar_to_save][current_step] += len(
                max(nx.connected_components(g), key=len))
        else:
            gs_up_to_stepcount[thetastar_to_save][current_step] = len(
                max(nx.connected_components(g), key=len))

    if len(unique_encounters) > 0:
        thetastar_to_save = get_thetastar(bee_array[0].thetastar)
        if current_step in unique_encounters_up_to_stepcount[thetastar_to_save]:
            unique_encounters_up_to_stepcount[thetastar_to_save][current_step] += len(unique_encounters)
        else:
            unique_encounters_up_to_stepcount[thetastar_to_save][current_step] = len(unique_encounters)

    if len(encounters_at_this_step) + running_total > 0:
        thetastar_to_save = get_thetastar(bee_array[0].thetastar)
        if current_step in total_encounters_up_to_stepcount[thetastar_to_save]:
            total_encounters_up_to_stepcount[thetastar_to_save][current_step] += len(
                encounters_at_this_step) + running_total
        else:
            total_encounters_up_to_stepcount[thetastar_to_save][current_step] = len(
                encounters_at_this_step) + running_total

    lock.release()
    return len(encounters_at_this_step)


###########################################################
#                   RANDOM WALK STUFF                     #
###########################################################


def populate_cell(all_paths, bee_number, step_i, cell_length, cells, bee_array):
    x_coord = all_paths[bee_number][step_i][0]
    y_coord = all_paths[bee_number][step_i][1]
    x_to_save = -1
    y_to_save = -1
    for col in range(0, int(NUM_CELLS_PER_ROW)):
        if x_coord == 0:
            x_to_save = 0
        lower_bound = col * cell_length
        upper_bound = (col + 1) * cell_length
        if lower_bound < x_coord <= upper_bound:
            x_to_save = col
            # found the right x coordinate
    for row in range(0, int(NUM_CELLS_PER_ROW)):
        if y_coord == 0:
            y_to_save = 0
        lower_bound = row * cell_length
        upper_bound = (row + 1) * cell_length
        if lower_bound < y_coord <= row + 1 * upper_bound:
            # found the right x, y coordinate
            y_to_save = row
    if x_to_save != -1 and y_to_save != -1:
        index = (y_to_save * NUM_CELLS_PER_ROW) + x_to_save
        if not bee_array[bee_number].placed:
            cells[index]['occupants'].append(bee_number)
            bee_array[bee_number].placed = True
            cells[index]['neighbor_indices'] = get_adjacent_cells(x_to_save, y_to_save)
    else:
        print "Something went wrong! x_coord = {}, y_coord = {}".format(x_coord, y_coord)


def random_walk(all_paths, bee_array, n, step_count, tracking_food, thread_name, unique_encounters):
    first_thread = False
    converged = False
    # set up data structures, decide whether to track food or not
    g = nx.Graph()
    ccs_over_time = {}
    num_encounters = 0
    cells = dict()
    cell_length = n / NUM_CELLS_PER_ROW
    present_ths = get_thetastar(bee_array[0].thetastar)
    nx_output_dir = "networks/networks_thetastar_{}_{}x{}_{}steps".format(present_ths, side_length,
                                                                          side_length, COUNTS)
    mkdir_p(nx_output_dir)
    if thread_name == 'Thread-0':
        first_thread = True
    for cell_num in range(0, NUM_CELLS):
        # for each cell, initialize a list that will hold bee numbers
        cells[cell_num] = {'neighbor_indices': [], 'occupants': []}

    ####################################
    #
    #   for each bee, set their initial position and initial cell locations
    #   then, for every step:
    #       for every bee:
    #          if the bee is not a donor or a receiver, it is not engaged in trophallaxis, so it moves
    #          along the random walk path corresponding to its theta*
    #              1. calculate new position based on old position and angle
    #              2. update position and angle
    #              3. populate cell lists with new bees
    #          if the bee IS a donor or a receiver, it is in trophallaxis, its position is unchanged
    #              1. decrement the steps waiting
    #                 a) if this hits zero, the bees re enter movement pool
    #              2. populate cell lists
    #              3. feed
    #
    ######################################

    for bee in bee_array:
        g.add_node(bee.number)
        all_paths[bee.number][0] = (bee.positionx[0], bee.positiony[0], [])
        populate_cell(all_paths, bee.number, 0, cell_length, cells, bee_array)
    #if first_thread:
        #nx.write_gml(g, NETWORK_STRING.format(nx_output_dir, 0))
    for step_i in range(1, step_count):
        ccs_over_time[step_i] = {}
        print step_i
        counter = 0
        avg_steps_since_encounter_at_this_timestep = 0
        avg_steps_since_unique_encounter_at_this_timestep = 0
        # move em around

        for bee in bee_array:
            if first_thread:
                cc_nodes = bee.agents_in_connected_component
                if len(cc_nodes) > 1:
                    if not any(cc_nodes) in ccs_over_time[step_i].values():
                        ccs_over_time[step_i][counter] = cc_nodes
                        counter = counter + 1

            if bee.donor is None and bee.receiver is None:
                theta = bee.thetastar[random.randint(0, THETASTARRANGE - 1)]
                bee.direction[step_i] = bee.direction[step_i - 1] + theta
                bee.positionx[step_i] = bee.positionx[step_i - 1] + vel * math.cos(bee.direction[step_i])

                if bee.positionx[step_i] > n:
                    bee.positionx[step_i] = bee.positionx[step_i] - n
                if bee.positionx[step_i] < 0:
                    bee.positionx[step_i] += n

                bee.positiony[step_i] = bee.positiony[step_i - 1] + vel * math.sin(bee.direction[step_i])

                if bee.positiony[step_i] > n:
                    bee.positiony[step_i] = bee.positiony[step_i] - n
                if bee.positiony[step_i] < 0:
                    bee.positiony[step_i] += n

                cc_nodes = bee.agents_in_connected_component
                all_paths[bee.number][step_i] = (bee.positionx[step_i], bee.positiony[step_i], cc_nodes)
                populate_cell(all_paths, bee.number, step_i, cell_length, cells, bee_array)
                bee.placed = False
            else:
                # if they are engaged in trophallaxis, positionx & positiony & direction remain constant
                bee.direction[step_i] = bee.direction[step_i - 1]
                bee.positionx[step_i] = bee.positionx[step_i - 1]
                bee.positiony[step_i] = bee.positiony[step_i - 1]
                populate_cell(all_paths, bee.number, step_i, cell_length, cells, bee_array)
                bee.placed = False
                bee.steps_to_wait -= 1
                if bee.steps_to_wait == 0:
                    bee.donor = None
                    bee.receiver = None
            
            for index in range(len(bee.step_count_of_last_encounter) - 1):
                diff = bee.step_count_of_last_encounter[index + 1] - bee.step_count_of_last_encounter[
                    index]
                avg_steps_since_encounter_at_this_timestep += diff
            for index in range(len(bee.step_count_of_last_unique_encounter) - 1):
                u_diff = bee.step_count_of_last_unique_encounter[index + 1] - \
                         bee.step_count_of_last_unique_encounter[index]
                avg_steps_since_unique_encounter_at_this_timestep += u_diff

#        if step_i in network_steps and first_thread:
            # if one of the steps we want to capture, write it for viz
            #nx.write_gml(g, NETWORK_STRING.format(nx_output_dir, step_i))
        # check that all bees are accounted for
        occupant_master_list = []
        for cell_num in range(0, NUM_CELLS):
            occupant_master_list.extend(cells[cell_num]['occupants'])
        assert len(occupant_master_list) == COUNTS

        # engage (set up the bees for trophallaxis in a subsequent step and track/count encounters)
        running_total = num_encounters
        num_encounters += engage(tracking_food, all_paths, g, step_i, n, bee_array, cells, thread_name,
                                 running_total, unique_encounters)
        feed(bee_array)
        # reset cell lists
        for cell_num in range(0, NUM_CELLS):
            cells[cell_num]['occupants'] = []

        # calculate steps between encounters at this timestep
        if num_encounters != 0:
            avg_steps_since_encounter_at_this_timestep = avg_steps_since_encounter_at_this_timestep / num_encounters

        if len(unique_encounters) > 0:
            avg_steps_since_unique_encounter_at_this_timestep = avg_steps_since_unique_encounter_at_this_timestep / len(
                unique_encounters)

        lock.acquire()
        if step_i in steps_to_encounter_over_time[present_ths]:
            steps_to_encounter_over_time[present_ths][
                step_i] += avg_steps_since_encounter_at_this_timestep
        else:
            steps_to_encounter_over_time[present_ths][
                step_i] = avg_steps_since_encounter_at_this_timestep
        if step_i in steps_to_unique_encounter_over_time[present_ths]:
            steps_to_unique_encounter_over_time[present_ths][
                step_i] += avg_steps_since_unique_encounter_at_this_timestep
        else:
            steps_to_unique_encounter_over_time[present_ths][
                step_i] = avg_steps_since_unique_encounter_at_this_timestep
        lock.release()

        # if feeding was happening, set up the food distribution tracking
        if tracking_food:
            ths = get_thetastar(bee_array[0].thetastar)
            food_levels = [bee.food_level for bee in bee_array]
            variance = np.var(food_levels)
            # print "Individual variance in food level at stepcount {}: {}".format(step_i, variance)
            fed_bees = [bee for bee in bee_array if bee.food_level > FOOD_THRESHOLD]
            num_of_fed_bees = len(fed_bees)
            lock.acquire()
            if step_i in food_distribution_vs_time[ths]:
                food_distribution_vs_time[ths][step_i] += num_of_fed_bees
            else:
                food_distribution_vs_time[ths][step_i] = num_of_fed_bees
            if num_of_fed_bees == COUNTS and not converged:
                print "Thetastar {} converged because all bees are fed at step {}".format(ths, step_i)
                #nx.write_gml(g, "{}/network_convergence_all_fed_step_{}.gml".format(nx_output_dir, step_i))
                convergence_times[thread_name][ths] = step_i
                networks_per_thread[thread_name][ths] = g
                # write me!
                unique_encounters_per_thread[thread_name][ths] = len(unique_encounters)
                converged = True
            if variance < VARIANCE_THRESHOLD and not converged:
                print "Thetastar {} converged due to variance at step {}!".format(ths, step_i)
#                nx.write_gml(g, "{}/network_convergence_variance_step_{}.gml".format(nx_output_dir, step_i))
                convergence_times[thread_name][ths] = step_i
                networks_per_thread[thread_name][ths] = g
                # write me!
                unique_encounters_per_thread[thread_name][ths] = len(unique_encounters)
                converged = True
            lock.release()
        else:
            ths = get_thetastar(bee_array[0].thetastar)
            if len(max(nx.connected_component_subgraphs(g))) == COUNTS and not converged:
                convergence_times[thread_name][ths] = step_i
                unique_encounters_per_thread[thread_name][ths] = len(unique_encounters)
                print "Thetastar {} converged due to connected component size at step {}!".format(ths, step_i)
                converged = True
#        if converged and tracking_food:
#            return num_encounters
    if first_thread:
        pos_over_time = {}
        for step in ccs_over_time.keys():
            pos_over_time[step] = {}
            dict_of_ccs = ccs_over_time[step]
            for num in dict_of_ccs.keys():
                list_of_ccs_positions = []
                cc_positions = []
                for b in dict_of_ccs[num]:
                    positionx = bee_array[b].positionx[step]
                    positiony = bee_array[b].positiony[step]
                    cc_positions.append((positionx, positiony))
                list_of_ccs_positions.extend(cc_positions)
                pos_over_time[step][num] = list_of_ccs_positions
        # print pos_over_time

        with open("bee_data_{}.json".format(get_thetastar(bee_array[0].thetastar)), "w") as fp:
            json.dump(pos_over_time, fp)

    lock.acquire()
    networks_per_thread[thread_name][present_ths] = g
    convergence_times[thread_name][present_ths] = STEPS
    lock.release()
    return num_encounters


###########################################################
#                       DRIVER                            #
###########################################################


def run_everything(is_tracking_food, thread_name):
    convergence_times[thread_name] = {}
    networks_per_thread[thread_name] = {}
    unique_encounters_per_thread[thread_name] = collections.OrderedDict()
    # set up food distribution tracking if specified
    if is_tracking_food:
        lock.acquire()
        for ts in THETASTARS:
            thetastar = get_thetastar(ts)
            food_distribution_vs_time[thetastar] = collections.OrderedDict()
        lock.release()

    # thread by trial! this function is called by each thread and results are written to the same dict
    for thetastar in THETASTARS:
        thetastar_range = get_thetastar(thetastar)
        print "Thetastar: {} in thread {}\n".format(thetastar_range, thread_name)

        # initialize the array of paths of shape bees x steps
        all_paths = np.zeros((COUNTS, STEPS), dtype=(object, 3))

        # initialize vector of agents and list of unique encounters
        unique_encounters = []
        bee_array = []

        # initialize along a lattice
        init_positionsx = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                      int(math.sqrt(COUNTS)))
        init_positionsy = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                      int(math.sqrt(COUNTS)))
        x, y = np.meshgrid(init_positionsx, init_positionsy)

        # initialize all bees
        for j in range(0, COUNTS):
            if random.randint(0, 100) < PCT_INITIALLY_CARRYING_FOOD:
                initially_fed = 1
            else:
                initially_fed = 0
            bee_array.append(Bee(j, thetastar, x.flatten(), y.flatten(), side_length, STEPS, 0,
                                 initially_fed))

        num_encounters = random_walk(all_paths, bee_array, side_length, STEPS, is_tracking_food,
                                     thread_name, unique_encounters)
        steps_since_encounter = 0
        steps_since_unique_encounter = 0
        for bee in bee_array:
            for index in range(len(bee.step_count_of_last_encounter) - 1):
                diff = bee.step_count_of_last_encounter[index + 1] - bee.step_count_of_last_encounter[
                    index]
                steps_since_encounter += diff
            for index in range(len(bee.step_count_of_last_unique_encounter) - 1):
                u_diff = bee.step_count_of_last_unique_encounter[index + 1] - bee.step_count_of_last_unique_encounter[index]
                steps_since_unique_encounter += u_diff

        avg_steps_between_encounters = 0
        if num_encounters == 0:
            print "No encounters!"
        else:
            avg_steps_between_encounters += steps_since_encounter / num_encounters
            print "Avg steps between encounters for thread {}: {}".format(thread_name,
                                                                          avg_steps_between_encounters)

        avg_steps_between_unique_encounters = 0
        if len(unique_encounters) > 0:
            avg_steps_between_unique_encounters += steps_since_unique_encounter / len(unique_encounters)
            print "Avg steps between unique encounters for thread {}: {}".format(thread_name,
                                                                                 avg_steps_between_unique_encounters)
        else:
            print "No unique encounters!"

        lock.acquire()
        if thetastar_range in steps_to_encounter:
            steps_to_encounter[thetastar_range][0] += avg_steps_between_encounters
            steps_to_encounter[thetastar_range][1] += avg_steps_between_unique_encounters
        else:
            steps_to_encounter[thetastar_range] = [avg_steps_between_encounters,
                                                   avg_steps_between_unique_encounters]
        lock.release()
        if thread_name == "Thread-0":
            print "writing"
            np.save('{}_range_{}_stepsfor_{}_agentsin_{}x{}box_{}'.format(thetastar_range, STEPS, COUNTS,
                                                                 side_length,
                                                                 side_length, num_encounters),
               all_paths)


def main():
    # parser = argparse.ArgumentParser(description='Run a 2D correlated random walk simulation.')
    # parser.add_argument('--count', help='Number of agents')
    # parser.add_argument('--steps', help='Number of steps')
    # parser.add_argument('--side_length', help='Side length of the system')
    # parser.add_argument('--food', help='Is there food in the system?')
    # args = parser.parse_args()
    # print args

    initialize_result_dictionaries()
    activate_food = False
    if sys.argv[1]:
        if str.lower(sys.argv[1]) == 'food':
            activate_food = True

    threads = []
    for trial in range(0, NUM_TRIALS):
        thread = BeeThread(activate_food, trial, "Thread-{}".format(trial), trial)
        threads.append(thread)
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    avg_convergence_times = setup_results(activate_food)
    write_data(activate_food, avg_convergence_times)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print "--- %s seconds ---" % (time.time() - start_time)
