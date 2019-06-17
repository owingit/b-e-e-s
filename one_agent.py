from __future__ import division
import matplotlib.pyplot as plt
import collections
import itertools
import networkx as nx
from scipy.optimize import curve_fit

import sys
import threading
import time

import random
import numpy as np
import math

import json

COUNTS = 1024
side_length = 716
STEPS = 5000
THETASTARRANGE = 100
THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), THETASTARRANGE) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]
THETASTARS.extend([np.linspace(0, 0, THETASTARRANGE)])
NUM_TRIALS = 100
INITIAL_DIRECTION = np.linspace(-math.pi, math.pi, THETASTARRANGE)
vel = 1.0

class Bee:
    def __init__(self, i, tstar, x_positions, y_positions, n, steps):
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
	self.positionx[0] = random.randint(0, n)
        self.positiony[0] = random.randint(0, n)
        self.direction = np.zeros(steps)
        self.direction[0] = INITIAL_DIRECTION[random.randint(0, THETASTARRANGE - 1)]
        self.thetastar = tstar
        self.name = "Bee #{}".format(i)
        self.number = i
	self.displacement = 0

def get_thetastar(thetastar_range):
    return thetastar_range[-1] - thetastar_range[0]

def random_walk(all_paths, bee_array, side_length, steps):
     for bee in bee_array:
        all_paths[bee.number][0] = (bee.positionx[0], bee.positiony[0], 0, [])
     for step_i in range(1, steps):
        for bee in bee_array:
            theta = bee.thetastar[random.randint(0, THETASTARRANGE - 1)]
            bee.direction[step_i] = bee.direction[step_i - 1] + theta
            bee.positionx[step_i] = bee.positionx[step_i - 1] + vel * math.cos(bee.direction[step_i])
            if bee.positionx[step_i] > side_length:
                bee.positionx[step_i] = bee.positionx[step_i] - side_length
            if bee.positionx[step_i] < 0:
                bee.positionx[step_i] += side_length

            bee.positiony[step_i] = bee.positiony[step_i - 1] + vel * math.sin(bee.direction[step_i])

            if bee.positiony[step_i] > side_length:
                bee.positiony[step_i] = bee.positiony[step_i] - side_length
            if bee.positiony[step_i] < 0:
               bee.positiony[step_i] += side_length
		
	    displacement = np.sqrt((bee.positionx[step_i] - bee.positionx[0])**2 + (bee.positiony[step_i] - bee.positiony[0])**2)
            if displacement >= side_length / np.sqrt(COUNTS):
                print step_i
        	return step_i

def func(x, a, b, c):
    return a * np.exp(-b * map(float, x)) + c


def run():
    data = {}
    for thetastar in THETASTARS:
        thetastar_range = get_thetastar(thetastar)
        data[thetastar_range] = 0    
    for trial in range(0, NUM_TRIALS+1):
        for thetastar in THETASTARS:
            thetastar_range = get_thetastar(thetastar)
	    print thetastar_range

            # initialize the array of paths of shape bees x steps
            all_paths = np.zeros((COUNTS, STEPS), dtype=(object, 4))

            # initialize vector of agents and list of unique encounters
            bee_array = []

            # initialize along a lattice
            init_positionsx = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                          int(math.sqrt(COUNTS)))
            init_positionsy = np.linspace(0, side_length - (side_length / math.sqrt(COUNTS) + 1),
                                      int(math.sqrt(COUNTS)))
            x, y = np.meshgrid(init_positionsx, init_positionsy)

            # initialize all bees
            bee_array.append(Bee(1, thetastar, x.flatten(), y.flatten(), side_length, STEPS))
            step_of_optimal_displacement = random_walk(all_paths, bee_array, side_length, STEPS)
            if data.get(thetastar_range):
	        data[thetastar_range] += step_of_optimal_displacement
	    else:
		data[thetastar_range] = step_of_optimal_displacement

    for key in data.keys():
	data[key] = data[key] / NUM_TRIALS

    plt.scatter(data.keys(), data.values())
    plt.xlabel('Thetastar')
    plt.ylabel('Steps')
    plt.title('Theta* vs. step count to reach {} displacement, {}x{} arena with {} agents'.format(side_length / np.sqrt(COUNTS), side_length, side_length, COUNTS))
#    popt, pcov = curve_fit(func,  data.keys(),  data.values())    
#    plt.plot(data.keys(), func(data.keys(), *popt), 'r-')
    plt.show()
def main():
    run()


if __name__ == "__main__":
    main()
