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

# initialize some data structures
GS = []
unique_encounters_up_to_stepcount = collections.OrderedDict()
total_encounters_up_to_stepcount = collections.OrderedDict()
gs_up_to_stepcount = collections.OrderedDict()

# constants
COUNTS = int(input("How many agents? (Enter a perfect square, please :P) "))
STEPS = int(input("How many steps?"))
ALL_PATHS = np.zeros((COUNTS, STEPS), dtype=(float, 2))
NS = [25, 50, 75, 100, 150, 200, 250, 500]
THETASTARRANGE = 50
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), THETASTARRANGE) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]
INITIAL_DIRECTION = np.linspace(-math.pi, math.pi, THETASTARRANGE)
NUM_TRIALS = 10  # 50
step_counts = [int(STEPS / 10), int(STEPS / 8), int(STEPS / 5), int(STEPS / 3), int(STEPS / 2),
               STEPS - 1]
vel = 1.0  # step size, or velocity
PCT_INITIALLY_CARRYING_FOOD = 10
FOOD_TRANSFER_RATE = 10  # units / timestep
FOOD_THRESHOLD = 5
NUM_CELLS = 25
NUM_CELLS_PER_ROW = math.sqrt(NUM_CELLS)
side_length = int(input("How long is each side of the arena?"))

food_distribution_vs_time = collections.OrderedDict()
for ts in THETASTARS:
    thetastar = ts[-1] - ts[0]
    food_distribution_vs_time[thetastar] = collections.OrderedDict()
# plot_enc_vs_thetastar = collections.OrderedDict()
# plot_unique_vs_thetastar = collections.OrderedDict()
steps_to_encounter = collections.OrderedDict()

# full bees should start near each other sorta done
# do they change orientation upon encounters? and if so, do they change back? or continue from the new trajectory?
# add food sources of various qualities that influence food transfer time / rate
# donor vs. expecter, where donors are stationary and expecters are mobile
# change the boundary conditions to allow for non-periodic boundary conditions

# Bee object


class BeeThread(threading.Thread):
    def __init__(self, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.counter = counter

    def run(self):
        print "Starting trial" + self.name
        run_everything()
        print "Exiting trial" + self.name


class Bee:
    def __init__(self, i, tstar, x_positions, y_positions, n, steps, r_or_u, initially_fed):
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if r_or_u == 1:
            if initially_fed:
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
        # self.food_transfer_rate = 10  # units / timestep
        self.donor = None
        self.receiver = None
        self.steps_to_wait = 0
        self.placed = False
        self.step_count_of_last_encounter = [0]
        self.step_count_of_last_unique_encounter = [0]


def write_data(steps_between_data, side_length):
    json_msg = '_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(COUNTS, side_length, side_length,
                                                                     STEPS)
    with open('steps' + json_msg, 'w') as fp:
        json.dump(steps_between_data.items(), fp, sort_keys=True)
    with open('unique_encounters_up_to_stepcount' + json_msg, 'w') as fp4:
        json.dump(unique_encounters_up_to_stepcount.items(), fp4, sort_keys=True)
    with open('total_encounters_up_to_stepcount' + json_msg, 'w') as fp5:
        json.dump(total_encounters_up_to_stepcount.items(), fp5, sort_keys=True)
    with open('gs' + json_msg, 'w') as fp6:
        json.dump(gs_up_to_stepcount.items(), fp6, sort_keys=True)


def setup_results():
    for key in gs_up_to_stepcount.keys():
        for ccsize in gs_up_to_stepcount[key].keys():
            gs_up_to_stepcount[key][ccsize] = gs_up_to_stepcount[key][ccsize] / (NUM_TRIALS)
        for sc in step_counts:
            total_encounters_up_to_stepcount[key][sc] = total_encounters_up_to_stepcount[key][sc] // NUM_TRIALS
            unique_encounters_up_to_stepcount[key][sc] = unique_encounters_up_to_stepcount[key][sc] // NUM_TRIALS
            if total_encounters_up_to_stepcount[key][sc] > 0:
                unique_encounters_up_to_stepcount[key][sc] = (unique_encounters_up_to_stepcount[key][sc] /
                                                              total_encounters_up_to_stepcount[key][sc]) * 100
            else:
                unique_encounters_up_to_stepcount[key][sc] = 0


def initialize_result_dictionaries():
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        unique_encounters_up_to_stepcount[thetastar] = collections.OrderedDict()
        total_encounters_up_to_stepcount[thetastar] = collections.OrderedDict()
        gs_up_to_stepcount[thetastar] = collections.OrderedDict()


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


def plot_food_distribution_vs_time(distribution_dict, side_length):
    for ts in THETASTARS:
        thetastar = ts[-1] - ts[0]
        distribution_dict[thetastar] = {int(key): val for key, val in
                                        distribution_dict[thetastar].items()}
        plt.plot(sorted(distribution_dict[thetastar].keys()), distribution_dict[thetastar].values(),
                 label='Thetastar: {}'.format(thetastar))
    plt.xlabel('Step')
    plt.ylabel('Number of fed individuals (threshold = {} units of food'.format(FOOD_THRESHOLD))
    plt.legend()
    plt.title('Number of fed individuals vs. time for {} agents in a {}x{} arena (n={})'.format(COUNTS,
                                                                                                side_length,
                                                                                                side_length,
                                                                                                NUM_TRIALS))
    plt.show()


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
            print "Bee {} food level {} feeding bee {} food level {}".format(donor,
                                                                             bee_array[donor].food_level,
                                                                             receiver, bee_array[
                                                                                 receiver].food_level)
            bee_array[donor].food_level -= FOOD_TRANSFER_RATE
            bee_array[receiver].food_level += FOOD_TRANSFER_RATE
            seen_donors.append(donor)


def engage(g, current_step, n, bee_array, cells, unique_encounters):
    print "Current step: {}\n, Cell array: {}\n".format(current_step, cells)
    encounters = []
    exclude_cells = []
    for cell in cells:
        locations_to_check = {}
        neighbor_bees_to_cell = []
        bees_in_cell = cells[cell]['occupants']
        for index in cells[cell]['neighbor_indices']:
            if index not in exclude_cells:
                neighbor_bees_to_cell.extend(cells[index]['occupants'])
        # print "Cell #{}, bees in cell: {}, neighbor bees: {}".format(cell, bees_in_cell,
        #                                                              neighbor_bees_to_cell)
        for bee in bees_in_cell:
            locations_to_check[bee] = tuple(ALL_PATHS[bee][current_step])
        for bee in neighbor_bees_to_cell:
            locations_to_check[bee] = tuple(ALL_PATHS[bee][current_step])
        for (a, ordered_pair_a), (b, ordered_pair_b) in itertools.combinations(
                locations_to_check.items(), 2):
            # a, b are bee numbers
            modified = False
            # correct for periodic boundary conditions, then return values back to what they were
            # when >2 non-donors meet, what happens
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
                if (a, b) not in encounters and (b, a) not in encounters:
                    encounters.append((a, b))
                    if g.has_edge(a, b) or g.has_edge(b, a):
                        g.edges[a, b]['weight'] += 1
                    else:
                        g.add_edge(a, b, weight=1)
                    if a >= COUNTS or b >= COUNTS:
                        continue
                    else:
                        bee_array[a].step_count_of_last_encounter.append(current_step)
                        bee_array[b].step_count_of_last_encounter.append(current_step)
                    if (a, b) not in unique_encounters and (b, a) not in unique_encounters:
                        if a >= COUNTS or b >= COUNTS:
                            continue
                        else:
                            bee_array[a].step_count_of_last_unique_encounter.append(current_step)
                            bee_array[b].step_count_of_last_unique_encounter.append(current_step)
                        unique_encounters.append((a, b))

            if math.hypot(ordered_pair_b[0] - ordered_pair_a[0],
                          ordered_pair_b[1] - ordered_pair_a[1]) <= vel and not modified:
                if (a, b) not in encounters and (b, a) not in encounters:
                    encounters.append((a, b))
                    if g.has_edge(a, b) or g.has_edge(b, a):
                        g.edges[a, b]['weight'] += 1
                    else:
                        g.add_edge(a, b, weight=1)
                    if a >= COUNTS or b >= COUNTS:
                        continue
                    else:
                        bee_array[a].step_count_of_last_encounter.append(current_step)
                        bee_array[b].step_count_of_last_encounter.append(current_step)
                    if (a, b) not in unique_encounters and (b, a) not in unique_encounters:
                        if a >= COUNTS or b >= COUNTS:
                            continue
                        else:
                            bee_array[a].step_count_of_last_unique_encounter.append(current_step)
                            bee_array[b].step_count_of_last_unique_encounter.append(current_step)
                        unique_encounters.append((a, b))
        exclude_cells.append(cell)

    if len(g.edges()) > 0:
        thetastar_to_save = bee_array[0].thetastar[-1] - bee_array[0].thetastar[0]
        if current_step in gs_up_to_stepcount[thetastar_to_save] and current_step in step_counts:
            gs_up_to_stepcount[thetastar_to_save][current_step] += len(
                max(nx.connected_components(g), key=len))
        else:
            gs_up_to_stepcount[thetastar_to_save][current_step] = len(
                max(nx.connected_components(g), key=len))

    if len(unique_encounters) > 0:
        thetastar_to_save = bee_array[0].thetastar[-1] - bee_array[0].thetastar[0]
        if current_step in unique_encounters_up_to_stepcount[thetastar_to_save] and current_step in step_counts:
            unique_encounters_up_to_stepcount[thetastar_to_save][current_step] += len(unique_encounters)
        else:
            unique_encounters_up_to_stepcount[thetastar_to_save][current_step] = len(unique_encounters)

    if len(encounters) > 0:
        thetastar_to_save = bee_array[0].thetastar[-1] - bee_array[0].thetastar[0]
        if current_step in total_encounters_up_to_stepcount[thetastar_to_save] and current_step in step_counts:
            total_encounters_up_to_stepcount[thetastar_to_save][current_step] += len(encounters)
        else:
            total_encounters_up_to_stepcount[thetastar_to_save][current_step] = len(encounters)

    return len(encounters)
    # ALL_PATHS[bee_number][current_step] has the location to compare
    # 0   1  2  3  4
    # 5   6  7  8  9
    # 10 11 12 13 14
    # 15 16 17 18 19
    # 20 21 22 23 24
    # 0: 1, 4, 5, 6, 9, 20, 21, 24
    # 1: 0, 2, 5, 6, 7, 20, 21, 22
    # 2: 1, 3, 6, 7, 8, 21, 22, 23
    # 3: 2, 4, 7, 8, 9, 22, 23, 24
    # 4: 0, 3, 5, 8, 9, 20, 23, 24
    # 5: 0, 1, 4, 6, 9, 10, 11, 14


def populate_cell(bee_number, step_i, cell_length, cells, bee_array):
    x_coord = ALL_PATHS[bee_number][step_i][0]
    y_coord = ALL_PATHS[bee_number][step_i][1]
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


def random_walk(bee_array, n, step_count, unique_encounters):
    g = nx.Graph()
    num_encounters = 0
    cells = dict()
    cell_length = n / NUM_CELLS_PER_ROW
    for cell_num in range(0, NUM_CELLS):
        # for each cell, initialize a list that will hold bee numbers
        cells[cell_num] = {'neighbor_indices': [], 'occupants': []}
    for bee in bee_array:
        ALL_PATHS[bee.number][0] = (bee.positionx[0], bee.positiony[0])
        populate_cell(bee.number, 0, cell_length, cells, bee_array)
    for step_i in range(1, step_count):
        for bee in bee_array:
            g.add_node(bee.number)
            if bee.donor is None and bee.receiver is None:
                # print "Bee {} is moving at timestep {}".format(bee.number, step_i)
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

                ALL_PATHS[bee.number][step_i] = (bee.positionx[step_i], bee.positiony[step_i])
                populate_cell(bee.number, step_i, cell_length, cells, bee_array)
                bee.placed = False
            else:
                # print "Bee {} is a donor or a receiver at timestep {}".format(bee.number, step_i)
                # if they are engaged in trophallaxis, positionx & positiony & direction remain constant
                bee.direction[step_i] = bee.direction[step_i - 1]
                bee.positionx[step_i] = bee.positionx[step_i - 1]
                bee.positiony[step_i] = bee.positiony[step_i - 1]
                bee.steps_to_wait -= 1
                feed(bee_array)
                if bee.steps_to_wait == 0:
                    bee.donor = None
                    bee.receiver = None

        occupant_master_list = []
        for cell_num in range(0, NUM_CELLS):
            occupant_master_list.extend(cells[cell_num]['occupants'])
        assert len(occupant_master_list) == COUNTS
        num_encounters += engage(g, step_i, n, bee_array, cells, unique_encounters)
        for cell_num in range(0, NUM_CELLS):
            cells[cell_num]['occupants'] = []
        # fed_bees = [bee for bee in bee_array if bee.food_level > FOOD_THRESHOLD]
        # num_of_fed_bees = len(fed_bees)
        # if step_i in distribution_dict[thetastar]:
        #     distribution_dict[thetastar][step_i] += num_of_fed_bees
        # else:
        #     distribution_dict[thetastar][step_i] = num_of_fed_bees
    return num_encounters


def run_everything():
    # thread by trial!
    for thetastar in THETASTARS:
        unique_encounters = []
        bee_array = []
        thetastar_range = thetastar[THETASTARRANGE - 1] - thetastar[0]
        init_positionsx = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                      int(math.sqrt(COUNTS)))
        init_positionsy = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                      int(math.sqrt(COUNTS)))
        x, y = np.meshgrid(init_positionsx, init_positionsy)
        print "Thetastar: {}".format(thetastar_range)

        # initialize all bees
        for j in range(0, COUNTS):
            if random.randint(0, 100) < PCT_INITIALLY_CARRYING_FOOD:
                initially_fed = 1
                # print "Bee {} is initially fed".format(j)
            else:
                initially_fed = 0
            bee_array.append(Bee(j, thetastar, x.flatten(), y.flatten(), side_length, STEPS, 1,
                                 initially_fed))

        num_encounters = random_walk(bee_array, side_length, STEPS, unique_encounters)
        steps_since_encounter = 0
        steps_since_unique_encounter = 0
        for bee in bee_array:
            msg = "Bee number: {}, Encounter steps: {}, Unique encounter steps: {}".format(
                bee.number,
                bee.step_count_of_last_encounter,
                bee.step_count_of_last_unique_encounter)
            print msg
            for index in range(len(bee.step_count_of_last_encounter) - 1):
                diff = bee.step_count_of_last_encounter[index + 1] - \
                       bee.step_count_of_last_encounter[index]
                steps_since_encounter += diff
            for index in range(len(bee.step_count_of_last_unique_encounter) - 1):
                u_diff = bee.step_count_of_last_unique_encounter[index + 1] - \
                         bee.step_count_of_last_unique_encounter[index]
                steps_since_unique_encounter += u_diff

        avg_steps_between_encounters = 0
        if num_encounters == 0:
            print "No encounters!"
        else:
            avg_steps_between_encounters += steps_since_encounter / num_encounters
            print "Avg steps between encounters: {}".format(avg_steps_between_encounters)

        avg_steps_between_unique_encounters = 0
        if len(unique_encounters) > 0:
            avg_steps_between_unique_encounters += steps_since_unique_encounter / len(
                unique_encounters)
            print "Avg steps between unique encounters: {}".format(
                avg_steps_between_unique_encounters)
        else:
            print "No unique encounters!"

        if thetastar_range in steps_to_encounter:
            steps_to_encounter[thetastar_range][0] += avg_steps_between_encounters
            steps_to_encounter[thetastar_range][1] += avg_steps_between_unique_encounters
        else:
            steps_to_encounter[thetastar_range] = [avg_steps_between_encounters,
                                                   avg_steps_between_unique_encounters]
        # fed_counter = 0
        # unfed_counter = 0
        # for bee in bee_array:
        #     if bee.food_level > FOOD_THRESHOLD:
        #         fed_counter += 1
        #     else:
        #         unfed_counter += 1

        print "Num encounters: {}".format(num_encounters)


    for key in steps_to_encounter.keys():
        steps_to_encounter[key][0] = steps_to_encounter[key][0] / NUM_TRIALS
        steps_to_encounter[key][1] = steps_to_encounter[key][1] / NUM_TRIALS

    write_data(steps_to_encounter, side_length)


#
# for ts in food_distribution_vs_time.keys():
#     for step in food_distribution_vs_time[ts].keys():
#         food_distribution_vs_time[ts][step] = food_distribution_vs_time[ts][step] / NUM_TRIALS
# with open('fed_bee_distribution_{}x{}_{}agents_TO_PLOT.json', 'w') as fp:
#     json.dump(food_distribution_vs_time.items(), fp, sort_keys=True)

# plot_food_distribution_vs_time(food_distribution_vs_time, side_length)


def main():
    initialize_result_dictionaries()

    threads = []
    for trial in range(0, NUM_TRIALS):
        thread = BeeThread(trial, "Thread-{}".format(trial), trial)
        threads.append(thread)
    for th in threads:
        th.start()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print "--- %s seconds ---" % (time.time() - start_time)
