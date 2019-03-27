from __future__ import division
import matplotlib.pyplot as plt
import collections
import itertools
import networkx as nx

import sys

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
THETASTARRANGE = 20
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), THETASTARRANGE) for i in (1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 8, 12)]
INITIAL_DIRECTION = np.linspace(-math.pi, math.pi, THETASTARRANGE)
NUM_TRIALS = 10 # 50
step_counts = [int(STEPS / 10), int(STEPS / 8), int(STEPS / 5), int(STEPS / 3), int(STEPS / 2), STEPS-1]
vel = 1.0 # step size, or velocity
PCT_INITIALLY_CARRYING_FOOD = 10
FOOD_TRANSFER_RATE = 10  # units / timestep
FOOD_THRESHOLD = 5
# full bees should start near each other sorta done
# do they change orientation upon encounters? and if so, do they change back? or continue from the new trajectory?
# add food sources of various qualities that influence food transfer time / rate
# donor vs. expecter, where donors are stationary and expecters are mobile
# change the boundary conditions to allow for non-periodic boundary conditions

# Bee object


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
					self.positionx[0] = random.randint(n - 2*PCT_INITIALLY_CARRYING_FOOD, n - PCT_INITIALLY_CARRYING_FOOD)
					self.positiony[0] = random.randint(n - 2*PCT_INITIALLY_CARRYING_FOOD, n - PCT_INITIALLY_CARRYING_FOOD)
			else:
				self.positionx[0] = random.randint(0, n)
				self.positiony[0] = random.randint(0, n)
		else:
			self.positionx[0] = x_positions[i]
			self.positiony[0] = y_positions[i]
		self.direction = np.zeros(steps)
		self.direction[0] = INITIAL_DIRECTION[random.randint(0, THETASTARRANGE-1)]
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


def plot_food_distribution_vs_time(distribution_dict, side_length):
	for ts in THETASTARS:
		thetastar = ts[-1] - ts[0]
		distribution_dict[thetastar] = {int(key): val for key, val in distribution_dict[thetastar].items()}
		plt.plot(sorted(distribution_dict[thetastar].keys()), distribution_dict[thetastar].values(), label='Thetastar: {}'.format(thetastar))
	plt.xlabel('Step')
	plt.ylabel('Number of fed individuals (threshold = {} units of food'.format(FOOD_THRESHOLD))
	plt.legend()
	plt.title('Number of fed individuals vs. time for {} agents in a {}x{} arena (n={})'.format(COUNTS, side_length, side_length, NUM_TRIALS))
	plt.show()


def check_x_boundary(coords, n):
	a = False
	b = False
	if coords[0] > n-1:
		a = True
	if coords[0] < 1:
		b = True
	return a, b


def check_y_boundary(coords, n):
	a = False
	b = False
	if coords[1] > n-1:
		a = True
	if coords[1] < 1:
		b = True
	return a, b


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
			print "Bee {} food level {} feeding bee {} food level {}".format(donor, bee_array[donor].food_level, receiver, bee_array[receiver].food_level)
			bee_array[donor].food_level -= FOOD_TRANSFER_RATE
			bee_array[receiver].food_level += FOOD_TRANSFER_RATE
			seen_donors.append(donor)


def engage(current_step, n, bee_array):
	current_locations = {}
	for bee_number in range(0, COUNTS):
		current_locations[bee_number] = tuple(ALL_PATHS[bee_number][current_step])

	for (a, ordered_pair_a), (b, ordered_pair_b) in itertools.combinations(current_locations.items(), 2):
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
			# encounter
			if bee_array[a].food_level > bee_array[b].food_level:
				if bee_array[a].donor is None and bee_array[b].receiver is None:
					# print "Bee {}, food level {} and bee {}, food level {} entering trophallaxis at step {}".format(a, bee_array[a].food_level, b, bee_array[b].food_level, current_step)
					# trophallaxis encounter
					bee_array[a].donor = a
					bee_array[a].receiver = b
					bee_array[b].donor = a
					bee_array[b].receiver = b
					steps_while_feeding = (bee_array[a].food_level - bee_array[b].food_level) / FOOD_TRANSFER_RATE
					bee_array[a].steps_to_wait = steps_while_feeding
					bee_array[b].steps_to_wait = steps_while_feeding
			elif bee_array[a].food_level < bee_array[b].food_level:
				if bee_array[a].donor is None and bee_array[b].receiver is None:
					# print "Bee {}, food level {} and bee {}, food level {} entering trophallaxis at step {}".format(a, bee_array[a].food_level, b, bee_array[b].food_level, current_step)
					# trophallaxis encounter
					bee_array[a].donor = b
					bee_array[a].receiver = a
					bee_array[b].donor = b
					bee_array[b].receiver = a
					steps_while_feeding = (bee_array[b].food_level - bee_array[a].food_level) / FOOD_TRANSFER_RATE
					bee_array[a].steps_to_wait = steps_while_feeding
					bee_array[b].steps_to_wait = steps_while_feeding
			else:
				# nothing happens encounter, move on
				bee_array[a].donor = None
				bee_array[a].receiver = None
				bee_array[b].donor = None
				bee_array[b].receiver = None
				bee_array[a].steps_to_wait = 0
				bee_array[b].steps_to_wait = 0

		if math.hypot(ordered_pair_b[0] - ordered_pair_a[0], ordered_pair_b[1] - ordered_pair_a[1]) <= vel and not modified:
			# encounter
			if bee_array[a].food_level > bee_array[b].food_level:
				if bee_array[a].donor is None and bee_array[b].receiver is None:
					# print "Bee {}, food level {} and bee {}, food level {} entering trophallaxis at step {}".format(a, bee_array[a].food_level, b, bee_array[b].food_level, current_step)
					# trophallaxis encounter
					bee_array[a].donor = a
					bee_array[a].receiver = b
					bee_array[b].donor = a
					bee_array[b].receiver = b
					steps_while_feeding = (bee_array[a].food_level - bee_array[b].food_level) / FOOD_TRANSFER_RATE
					bee_array[a].steps_to_wait = steps_while_feeding
					bee_array[b].steps_to_wait = steps_while_feeding
			elif bee_array[a].food_level < bee_array[b].food_level:
				if bee_array[a].donor is None and bee_array[b].receiver is None:
					# print "Bee {}, food level {} and bee {}, food level {} entering trophallaxis at step {}".format(a, bee_array[a].food_level, b, bee_array[b].food_level, current_step)
					# trophallaxis encounter
					bee_array[a].donor = b
					bee_array[a].receiver = a
					bee_array[b].donor = b
					bee_array[b].receiver = a
					steps_while_feeding = (bee_array[b].food_level - bee_array[a].food_level) / FOOD_TRANSFER_RATE
					bee_array[a].steps_to_wait = steps_while_feeding
					bee_array[b].steps_to_wait = steps_while_feeding
			else:
				# nothing happens encounter, move on
				bee_array[a].donor = None
				bee_array[a].receiver = None
				bee_array[b].donor = None
				bee_array[b].receiver = None
				bee_array[a].steps_to_wait = 0
				bee_array[b].steps_to_wait = 0


def random_walk(bee_array, n, thetastar, step_count, distribution_dict):
	# g = nx.Graph()
	# num_bees = count
	# g_theta = bee_array[0].thetastar
	for bee in bee_array:
		ALL_PATHS[bee.number][0] = (bee.positionx[0], bee.positiony[0])
	for step_i in range(1, step_count):
		for bee in bee_array:
			if bee.donor is None and bee.receiver is None:
				# print "Bee {} is moving at timestep {}".format(bee.number, step_i)
				theta = bee.thetastar[random.randint(0, THETASTARRANGE - 1)]
				bee.direction[step_i] = bee.direction[step_i - 1] + theta
				bee.positionx[step_i] = bee.positionx[step_i - 1] + vel*math.cos(bee.direction[step_i])

				if bee.positionx[step_i] > n:
					bee.positionx[step_i] = bee.positionx[step_i] - n
				if bee.positionx[step_i] < 0:
					bee.positionx[step_i] += n

				bee.positiony[step_i] = bee.positiony[step_i - 1] + vel*math.sin(bee.direction[step_i])

				if bee.positiony[step_i] > n:
					bee.positiony[step_i] = bee.positiony[step_i] - n
				if bee.positiony[step_i] < 0:
					bee.positiony[step_i] += n

				ALL_PATHS[bee.number][step_i] = (bee.positionx[step_i], bee.positiony[step_i])
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
		engage(step_i, n, bee_array)
		fed_bees = [bee for bee in bee_array if bee.food_level > FOOD_THRESHOLD]
		num_of_fed_bees = len(fed_bees)
		if step_i in distribution_dict[thetastar]:
			distribution_dict[thetastar][step_i] += num_of_fed_bees
		else:
			distribution_dict[thetastar][step_i] = num_of_fed_bees


def main():
	random_or_uniform = sys.argv[1]
	rando = 0
	if str.lower(random_or_uniform) == 'u':
		rando = 0
	elif str.lower(random_or_uniform) == 'r':
		rando = 1

	food_distribution_vs_time = collections.OrderedDict()
	for ts in THETASTARS:
		thetastar = ts[-1] - ts[0]
		food_distribution_vs_time[thetastar] = collections.OrderedDict()

	side_length = int(input("How long is each side of the arena?"))
	for trial in range(0, NUM_TRIALS):
		print "Trial {}".format(trial)
		for thetastar in THETASTARS:
			bee_array = []
			thetastar_range = thetastar[THETASTARRANGE - 1] - thetastar[0]
			init_positionsx = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1), int(math.sqrt(COUNTS)))
			init_positionsy = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1), int(math.sqrt(COUNTS)))
			x, y = np.meshgrid(init_positionsx, init_positionsy)
			print "Thetastar: {}".format(thetastar_range)

			# initialize all bees
			for j in range(0, COUNTS):
				if random.randint(0, 100) < PCT_INITIALLY_CARRYING_FOOD:
					initially_fed = 1
					print "Bee {} is initially fed".format(j)
				else:
					initially_fed = 0
				bee_array.append(Bee(j, thetastar, x.flatten(), y.flatten(), side_length, STEPS, rando, initially_fed))

			random_walk(bee_array, side_length, thetastar_range, STEPS, food_distribution_vs_time)

	for ts in food_distribution_vs_time.keys():
		for step in food_distribution_vs_time[ts].keys():
			food_distribution_vs_time[ts][step] = food_distribution_vs_time[ts][step] / NUM_TRIALS
	with open('fed_bee_distribution_{}x{}_{}agents_TO_PLOT.json', 'w') as fp:
		json.dump(food_distribution_vs_time.items(), fp, sort_keys=True)
	plot_food_distribution_vs_time(food_distribution_vs_time, side_length)


if __name__ == "__main__":
	main()


