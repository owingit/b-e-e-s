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
NUM_TRIALS = 25 # 50
step_counts = [int(STEPS / 10), int(STEPS / 8), int(STEPS / 5), int(STEPS / 3), int(STEPS / 2), STEPS-1]
vel = 1.0 # step size, or velocity
# % starting as full
# full bees should start near each other
# do they change orientation upon encounters? and if so, do they change back? or continue from the new trajectory?
# wait time for trophallaxis
# add food sources of various qualities that influence food transfer time / rate
# donor vs. expecter, where donors are stationary and expecters are mobile
# change the boundary conditions to allow for non-periodic boundary conditions

# Bee object


class Bee:
	def __init__(self, i, tstar, x_positions, y_positions, n, steps, r_or_u):
		self.positionx = np.zeros(steps)
		self.positiony = np.zeros(steps)
		if r_or_u == 1:
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
		self.step_count_of_last_encounter = [0]
		self.step_count_of_last_unique_encounter = [0]
		# food level
		# food variance
		# food transfer rate (number of units / timestep)
		# food transfer time (time to sleep for each unit of food)
		# type (donor, receiver)


def initialize_result_dictionaries():
	for sc in step_counts:
		total_encounters_up_to_stepcount[sc] = collections.OrderedDict()
		unique_encounters_up_to_stepcount[sc] = collections.OrderedDict()
	for ts in THETASTARS:
		thetastar = ts[-1] - ts[0]
		gs_up_to_stepcount[thetastar] = collections.OrderedDict()


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


def plot_avg_steps(steps_to_encounter, r_or_u, side_length):
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'
	plt.plot(steps_to_encounter.keys(), steps_to_encounter.values())
	plt.xlabel('Thetastar')
	plt.ylabel('Avg steps between encounters')
	plt.title(
		'Avg number of steps between encounters and between unique encounters for {} {} distributed agents in {}x{} arena over {} steps (n={})'.format(COUNTS, random_string, side_length, side_length, STEPS, NUM_TRIALS))
	plt.show()


def plot_gs_with_time(side_length, r_or_u, count):
	print "in plot gs and time"
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'

	for ts in THETASTARS:
		thetastar = ts[-1] - ts[0]
		plt.plot(sorted(gs_up_to_stepcount[thetastar].keys()[:-1]), gs_up_to_stepcount[thetastar].values()[:-1], label='Thetastar: {}'.format(thetastar))
	plt.xlabel('Step')
	plt.ylabel('Size of the largest connected component')
	plt.legend()
	plt.title('Size of the largest connected component vs. time for {} {} distributed agents in a {}x{} arena (n={})'.format(count, random_string, side_length, side_length, NUM_TRIALS))
	plt.show()


def plot_gs(side_length, count, steps):
	print "in plot gs"
	i = 1
	for (G, G_theta) in GS:
		connected_component_sizes = dict.fromkeys(range(count+1), 0)
		list_connected_components = nx.connected_components(G)
		for cc in list_connected_components:
			connected_component_sizes[len(cc)] += 1

		thetastar = G_theta[THETASTARRANGE-1] - G_theta[0]
		plt.figure(i)
		plt.bar(connected_component_sizes.keys(), connected_component_sizes.values())
		plt.title("Distribution of connected components within the network of encounters between {} agents following {} steps in a {}x{} arena with {} thetastar".format(count, steps, side_length, side_length, thetastar))
		# edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
		# plt.figure(i)
		# plt.title("Connectivity after {} agents moved {} steps in a {}x{} arena with thetastar of {}".format(count, steps, side_length, side_length, thetastar))
		# nx.draw_networkx(G, node_size=100, edge_color=weights, edge_cmap=plt.cm.bwr, width=weights)
		i += 1	
	plt.show()


def plot_tstar_and_steps(side_length, r_or_u, count):
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'

	for sc in step_counts:
		plt.plot(unique_encounters_up_to_stepcount[sc].keys(), unique_encounters_up_to_stepcount[sc].values(), label='Step count: {}'.format(sc))
	plt.xlabel('Theta*: width of the range of possible angles')
	plt.ylabel('% Unique encounters')
	plt.legend()
	plt.title('Unique encounters vs. theta star of {} {} distributed across a wide range of step counts in a {}x{} arena'.format(count, random_string, side_length, side_length))
	plt.show()


def plot_tstar(plot_enc_vs_thetastar, plot_unique_vs_thetastar, side_length, r_or_u, count, step_count):
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'
	fig, ax1 = plt.subplots()
	ax1.bar(plot_enc_vs_thetastar.keys(), plot_enc_vs_thetastar.values(), 0.2)
	ax2 = ax1.twinx()
	ax2.plot(plot_unique_vs_thetastar.keys(), plot_unique_vs_thetastar.values(), color='orange')
	ax1.set_xlabel('Theta* : the width of the range of possible angles')
	ax1.set_ylabel('Num of encounters (whole system)')
	ax2.set_ylabel('% unique encounters (whole system)')
	plt.title('Number of encounters between {} {} distributed agents as a function of theta* with an arena size of {}x{} over {} steps'.format(count, random_string, side_length, side_length, step_count))
	plt.savefig('{}_agents_{}x{}_vary_thetastar.png'.format(count, side_length, side_length))
	# for thetastar in plot_unique_vs_thetastar.keys():
	# to_save = [step_count, thetastar, plot_unique_vs_thetastar[thetastar]]
	# np.savetxt('{}_steps-{}_agents-{}x{}-{}_angle_range'.format(step_count, count, side_length, side_length, thetastar), to_save)
	plt.show()


def plot_size_steps(r_or_u):
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'
	for sc in step_counts:
		plt.plot(unique_encounters_up_to_stepcount[sc].keys(), unique_encounters_up_to_stepcount[sc].values(), label='Step count: {}'.format(sc))
	plt.xlabel('N: Side length of the arena')
	plt.ylabel('% Unique encounters')
	plt.legend()
	plt.title('Unique encounters vs. side length across a wide range of step counts')
	plt.show()


def plot_size(plot_enc_vs_size, plot_unique_vs_size, theta_index, r_or_u, count, step_count):
	if r_or_u == 0:
		random_string = 'uniformly'
	else:
		random_string = 'randomly'
	print plot_enc_vs_size
	print plot_unique_vs_size
	fig, ax1 = plt.subplots()
	ax1.scatter(plot_enc_vs_size.keys(), plot_enc_vs_size.values())
	ax2 = ax1.twinx()
	ax2.plot(plot_unique_vs_size.keys(), plot_unique_vs_size.values(), color='orange')
	ax1.set_xlabel('Side length (N, for an NxN arena)')
	ax1.set_ylabel('Num of encounters (whole system)')
	ax2.set_ylabel('% unique encounters (whole system)')
	plt.title('Number of encounters between {} {} distributed agents as a function of N with a theta range between {} & {} over {} steps'.format(count, random_string, THETASTARS[theta_index][0], THETASTARS[theta_index][THETASTARRANGE-1], step_count))
	plt.savefig('{}_agents_piover6_vary_N.png'.format(count))
	plt.show()


def count_encounters(count, step_count, n, set_g, count_things, first_loop, g, g_theta, bee_array, uniques=None):
	thetastar = g_theta[-1] - g_theta[0]
	locations_at_timestep = {}
	num_encounters = 0
	for i in range(0, step_count):
		locations_at_timestep[i] = []
		for j in range(0, count):
			t = tuple(np.ndarray.tolist(ALL_PATHS[j][i]))
			locations_at_timestep[i].append((j, t))
	
	for ts in range(0, step_count):
		for (a, ordered_pair_a), (b, ordered_pair_b) in itertools.combinations(locations_at_timestep[ts], 2):
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
				if set_g:
					if g.has_edge(a, b) or g.has_edge(b,a):
						g.edges[a, b]['weight'] += 1
					else:
						g.add_edge(a, b, weight=1)
				if count_things and a >= count or b >= count:
					continue
				else:
					bee_array[a].step_count_of_last_encounter.append(ts)
					bee_array[b].step_count_of_last_encounter.append(ts)
				num_encounters += 1
				if (a, b) not in uniques and (b, a) not in uniques:
					if a >= count or b >= count:
						continue
					else:
						bee_array[a].step_count_of_last_unique_encounter.append(ts)
						bee_array[b].step_count_of_last_unique_encounter.append(ts)
					uniques.append((a, b))

			if math.hypot(ordered_pair_b[0] - ordered_pair_a[0], ordered_pair_b[1] - ordered_pair_a[1]) <= vel and not modified:
				if set_g:
					if g.has_edge(a, b) or g.has_edge(b, a):
						g.edges[a, b]['weight'] += 1
					else:
						g.add_edge(a, b, weight=1)

				if count_things and a >= count or b >= count:
					continue
				else:
					bee_array[a].step_count_of_last_encounter.append(ts)
					bee_array[b].step_count_of_last_encounter.append(ts)
				num_encounters += 1
				if (a, b) not in uniques and (b, a) not in uniques:
					if a >= count or b >= count:
						continue
					else:
						bee_array[a].step_count_of_last_unique_encounter.append(ts)
						bee_array[b].step_count_of_last_unique_encounter.append(ts)
					uniques.append((a, b))

		if len(g.edges()) > 0:
			if ts in gs_up_to_stepcount[thetastar]:
				gs_up_to_stepcount[thetastar][ts] += len(max(nx.connected_components(g), key=len))
			else:
				gs_up_to_stepcount[thetastar][ts] = len(max(nx.connected_components(g), key=len))
	if set_g and first_loop:
		GS.append((g, g_theta))
	return num_encounters


def random_walk(bee_array, n, count, step_count, trial_num, encountees=None):
	g = nx.Graph()
	g_theta = bee_array[0].thetastar
	num_encounters = 0
	first_loop = False
	for bee in bee_array:
		g.add_node(bee.number)
		# Initialize the numpy array tracking all bee paths. This has the shape MxN where M
		# is the bee count, N the number of steps. Each is a tuple of x,y positions; the entire
		# N length list represents the path of an individual
		ALL_PATHS[bee.number][0] = (bee.positionx[0], bee.positiony[0])
		for step_i in range(1, step_count):
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

		if trial_num == 0:
			first_loop = True

	num_encounters += count_encounters(count, step_count, n, True, True, first_loop, g, g_theta, bee_array, encountees)
	return num_encounters


def setup_results():
	for sc in step_counts:
		for key in total_encounters_up_to_stepcount[sc].keys():
			total_encounters_up_to_stepcount[sc][key] = total_encounters_up_to_stepcount[sc][key] // NUM_TRIALS
			unique_encounters_up_to_stepcount[sc][key] = unique_encounters_up_to_stepcount[sc][key] // NUM_TRIALS
			if total_encounters_up_to_stepcount[sc][key] > 0:
				unique_encounters_up_to_stepcount[sc][key] = (unique_encounters_up_to_stepcount[sc][key] / total_encounters_up_to_stepcount[sc][key]) * 100
			else:
				unique_encounters_up_to_stepcount[sc][key] = 0

	for key in gs_up_to_stepcount.keys():
		for ccsize in gs_up_to_stepcount[key].keys():
			gs_up_to_stepcount[key][ccsize] = gs_up_to_stepcount[key][ccsize] / (NUM_TRIALS)


def write_data(steps_between_data, e_data, u_e_data, side_length):
	json_msg = '_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(COUNTS, side_length, side_length, STEPS)
	with open('steps'+json_msg, 'w') as fp:
		json.dump(steps_between_data.items(), fp, sort_keys=True)
	with open('encounters'+json_msg, 'w') as fp2:
		json.dump(e_data.items(), fp2, sort_keys=True)
	with open('unique_encounters'+json_msg, 'w') as fp3:
		json.dump(u_e_data.items(), fp3, sort_keys=True)
	with open('unique_encounters_up_to_stepcount'+json_msg, 'w') as fp4:
		json.dump(unique_encounters_up_to_stepcount.items(), fp4, sort_keys=True)
	with open('total_encounters_up_to_stepcount'+json_msg, 'w') as fp5:
		json.dump(total_encounters_up_to_stepcount.items(), fp5, sort_keys=True)

	with open('gs'+json_msg, 'w') as fp6:
		json.dump(gs_up_to_stepcount.items(), fp6, sort_keys=True)


def main():
	random_or_uniform = sys.argv[1]
	save_data = sys.argv[2]
	rando = 0
	if str.lower(random_or_uniform) == 'u':
		rando = 0
	elif str.lower(random_or_uniform) == 'r':
		rando = 1

	# Determine which simulation to run
	which_vary = raw_input("What do you wish to vary: Theta* or N?")

	# initialize result dictionaries
	initialize_result_dictionaries()

	# TO DO: fix vary N
	if which_vary == 'N':
		plot_enc_vs_size = collections.OrderedDict()
		plot_unique_vs_size = collections.OrderedDict()
		theta_index = int(input("Enter theta* index:\n {0: pi/0.5, 1: pi, 2: pi/1.25, 3: pi/1.5, 4: pi/1.75, 5: pi/2, 6: pi/3, 7: pi/4, 8: pi/5, 9: pi/6, 10: pi/8, 11: pi/12} "))
		for trial in range(0, NUM_TRIALS):
			print "Trial {}".format(trial)
			for N in NS:
				# initialize the bee array
				bee_array = []
				print "N = {}".format(N)
				init_positionsx = np.linspace(0, N - (N / math.sqrt(COUNTS) + 1), int(math.sqrt(COUNTS)))
				init_positionsy = np.linspace(0, N - (N / math.sqrt(COUNTS) + 1), int(math.sqrt(COUNTS)))
				x, y = np.meshgrid(init_positionsx, init_positionsy)
				encountees = []
				num_encounters = 0

				for j in range(0, COUNTS):
					bee_array.append(Bee(j, THETASTARS[theta_index], x.flatten(), y.flatten(), N, STEPS, rando)) #thetastar = -pi/6 to pi/6

				num_encounters += random_walk(bee_array, N, COUNTS, STEPS, trial, encountees)

				for sc in step_counts:
					uniques = []
					limited_encounters = count_encounters(COUNTS, sc, N, False, False, False, nx.Graph(), 0, uniques)
					if N in unique_encounters_up_to_stepcount[sc]:
						unique_encounters_up_to_stepcount[sc][N] += len(uniques)
					else:
						unique_encounters_up_to_stepcount[sc][N] = len(uniques)
					if N in total_encounters_up_to_stepcount[sc]:
						total_encounters_up_to_stepcount[sc][N] += limited_encounters
					else:
						total_encounters_up_to_stepcount[sc][N] = limited_encounters

				if N in plot_unique_vs_size:
					plot_unique_vs_size[N] += len(encountees)
				else:
					plot_unique_vs_size[N] = len(encountees)
				
				if N in plot_enc_vs_size:
					plot_enc_vs_size[N] += num_encounters
				else:
					plot_enc_vs_size[N] = num_encounters

				# if save_data == 'Y':
					# np.save('{}_range_{}_stepsfor_{}_agentsin_{}x{}box_{}'.format(THETASTARS[theta_index], STEPS, COUNTS, N, N, num_encounters), ALL_PATHS)

		setup_results()

		for key in plot_enc_vs_size.keys():
			plot_enc_vs_size[key] = plot_enc_vs_size[key] // NUM_TRIALS
			plot_unique_vs_size[key] = plot_unique_vs_size[key] / NUM_TRIALS
			if plot_enc_vs_size[key] > 0:
				plot_unique_vs_size[key] = (plot_unique_vs_size[key] / plot_enc_vs_size[key]) * 100
			else:
				plot_unique_vs_size[key] = 0
		plot_size(plot_enc_vs_size, plot_unique_vs_size, theta_index, rando, COUNTS, STEPS)
		plot_size_steps(rando)

	elif which_vary == 'Theta*':
		plot_enc_vs_thetastar = collections.OrderedDict()
		plot_unique_vs_thetastar = collections.OrderedDict()
		steps_to_encounter = collections.OrderedDict()
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
				num_encounters = 0
				encountees = []
				
				# initialize all bees
				for j in range(0, COUNTS):
					bee_array.append(Bee(j, thetastar, x.flatten(), y.flatten(), side_length, STEPS, rando))

				num_encounters += random_walk(bee_array, side_length, COUNTS, STEPS, trial, encountees)

				steps_since_encounter = 0
				steps_since_unique_encounter = 0
				for bee in bee_array:
					msg = "Bee number: {}, Encounter steps: {}, Unique encounter steps: {}".format(
						bee.number,
						bee.step_count_of_last_encounter,
						bee.step_count_of_last_unique_encounter)
					print msg
					for index in range(len(bee.step_count_of_last_encounter) -1):
						diff = bee.step_count_of_last_encounter[index+1] - bee.step_count_of_last_encounter[index]
						steps_since_encounter += diff
					for index in range(len(bee.step_count_of_last_unique_encounter) -1):
						u_diff = bee.step_count_of_last_unique_encounter[index + 1] - bee.step_count_of_last_unique_encounter[index]
						steps_since_unique_encounter += u_diff

				avg_steps_between_encounters = 0
				if num_encounters == 0:
					print "No encounters!"
				else:
					avg_steps_between_encounters += steps_since_encounter / num_encounters
					print "Avg steps between encounters: {}".format(avg_steps_between_encounters)

				avg_steps_between_unique_encounters = 0
				if len(encountees) > 0:
					avg_steps_between_unique_encounters += steps_since_unique_encounter / len(encountees)
					print "Avg steps between unique encounters: {}".format(avg_steps_between_unique_encounters)
				else:
					print "No unique encounters!"

				if thetastar_range in steps_to_encounter:
					steps_to_encounter[thetastar_range][0] += avg_steps_between_encounters
					steps_to_encounter[thetastar_range][1] += avg_steps_between_unique_encounters
				else:
					steps_to_encounter[thetastar_range] = [avg_steps_between_encounters, avg_steps_between_unique_encounters]

				for sc in step_counts:
					uniques = []
					limited_encounters = count_encounters(COUNTS, sc, side_length, False, False, False, nx.Graph(), thetastar, bee_array, uniques)
					if thetastar_range in unique_encounters_up_to_stepcount[sc]:
						unique_encounters_up_to_stepcount[sc][thetastar_range] += len(uniques)	
					else:
						unique_encounters_up_to_stepcount[sc][thetastar_range] = len(uniques)
					if thetastar_range in total_encounters_up_to_stepcount[sc]:
						total_encounters_up_to_stepcount[sc][thetastar_range] += limited_encounters
					else:
						total_encounters_up_to_stepcount[sc][thetastar_range] = limited_encounters

				if thetastar_range in plot_unique_vs_thetastar:
					plot_unique_vs_thetastar[thetastar_range] += len(encountees)
				else:
					plot_unique_vs_thetastar[thetastar_range] = len(encountees)

				print num_encounters, " in thetastar loop"
				if thetastar_range in plot_enc_vs_thetastar:
					plot_enc_vs_thetastar[thetastar_range] += num_encounters
				else:
					plot_enc_vs_thetastar[thetastar_range] = num_encounters

				# if save_data == 'Y':
				#	np.save('{}_range_{}_stepsfor_{}_agentsin_{}x{}box_{}'.format(thetastar_range, STEPS, COUNTS, side_length, side_length, num_encounters), ALL_PATHS)

		setup_results()
		for key in steps_to_encounter.keys():
			steps_to_encounter[key][0] = steps_to_encounter[key][0] / NUM_TRIALS
			steps_to_encounter[key][1] = steps_to_encounter[key][1] / NUM_TRIALS

		for key in plot_enc_vs_thetastar.keys():
			plot_enc_vs_thetastar[key] = plot_enc_vs_thetastar[key] // NUM_TRIALS
			plot_unique_vs_thetastar[key] = plot_unique_vs_thetastar[key] / NUM_TRIALS
			if plot_enc_vs_thetastar[key] > 0:
				plot_unique_vs_thetastar[key] = (plot_unique_vs_thetastar[key] / plot_enc_vs_thetastar[key]) * 100
			else:
				plot_unique_vs_thetastar[key] = 0

		if save_data == 'Y':
			write_data(steps_to_encounter, plot_enc_vs_thetastar, plot_unique_vs_thetastar, side_length)
		else:
			plot_avg_steps(steps_to_encounter, rando, side_length)
			plot_tstar(plot_enc_vs_thetastar, plot_unique_vs_thetastar, side_length, rando, COUNTS, STEPS)
			plot_tstar_and_steps(side_length, rando, COUNTS)
			plot_gs(side_length, COUNTS, STEPS)
			plot_gs_with_time(side_length, rando, COUNTS)
	else: 
		print "Please enter either Theta* or N as your option."


if __name__ == "__main__":
	main()

