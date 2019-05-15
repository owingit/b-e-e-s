import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import collections
import json
import math


THETASTARS = [np.linspace(-(math.pi / i), (math.pi / i), 100) for i in
              (1, 1.5, 2, 3, 4, 6, 8, 12)]

counts = input("how many agents?")
side_length = input("how long were the sides?")
steps = input("how many steps?")
num_trials = input("how many trials?")
with open('gs_between_{}agents_{}x{}_{}steps_TO_PLOT.json'.format(
        counts, side_length, side_length, steps), 'r') as fp5:
    gs_up_to_stepcount_data = collections.OrderedDict(json.load(fp5))

# for ts in THETASTARS:
#     thetastar = ts[-1] - ts[0]
#     gs_up_to_stepcount_data[thetastar] = {int(key): val for key, val in gs_up_to_stepcount_data[thetastar].items()}
#     plt.plot(sorted(gs_up_to_stepcount_data[thetastar].keys()),
#                  gs_up_to_stepcount_data[thetastar].values(), label='Thetastar: {}'.format(thetastar))
# plt.xlabel('Step')
# plt.ylabel('Size of the largest connected component')
# plt.legend()
# plt.title('Size of the largest connected component vs. time for {} agents in a {}x{} arena (n={})'.format(
#     counts, side_length, side_length, num_trials))
# plt.show()
for ts in THETASTARS:
    thetastar = ts[-1] - ts[0]
    gs_up_to_stepcount_data[thetastar] = {float(key): float(val) for key, val in gs_up_to_stepcount_data[thetastar].items()}

a = float(np.random.exponential(size=1))
r = float(np.random.exponential(size=1))


# logistic function definition
def f(x, a, c, r):
    return c / (1. + (a * (np.exp(-r * x))))


for ts in THETASTARS:
    thetastar = ts[-1] - ts[0]
    l = len(gs_up_to_stepcount_data[thetastar].keys())
    x = np.linspace(1., float(l), l)
    y_model = f(x, a, counts, r)
    y = gs_up_to_stepcount_data[thetastar].values()
    (a_, c_, r_), _ = opt.curve_fit(f, x, y)
    y_fit = f(x, a_, c_, r_)
    print r_
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, y_fit, '--k', label='r = {}, a = {}'.format(r_, a_))
    ax.plot(x, y, 'o')
    plt.xlabel('Step')
    plt.ylabel('Size of the largest connected component')
    plt.title('Fitted logistic regression for the largest connected component over time for thetastar of {}'.format(thetastar))
    plt.legend()
    plt.show()