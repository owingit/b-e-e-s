import numpy as np
from sklearn import preprocessing
import scipy.optimize as opt
from scipy.stats import genlogistic
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

a = counts / 2
r = float(np.random.exponential(size=1))


# logistic function definition
def f(x, a, c, r):
    return c / (1. + (a * (np.exp(-r * np.asarray(x)))))

#
# def probability_distribution(r, mean, x):
#     return (np.exp((x*-1) - mean) / r) / r * (1 + (np.exp((x*-1) - mean) / r) **2)


for ts in THETASTARS:
    thetastar = ts[-1] - ts[0]
    l = len(gs_up_to_stepcount_data[thetastar].keys())
    x = np.linspace(1., float(l), l)
    y_model = f(x, a, counts, r)
    y = gs_up_to_stepcount_data[thetastar].values()
    print len(y)
    (a_, c_, r_), _ = opt.curve_fit(f, x, y, maxfev=1000, bounds=(0, [counts / 2, counts, 1]))
    y_fit = f(x, a_, c_, r_)
    print r_
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.plot(x, y_fit, '--k', label='r = {}, inflection_point = {}'.format(r_, (math.log(a_) / r_)))
    ax1.plot(x, y, 'o')
    plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(x, genlogistic.pdf(x, loc=math.log(a_) / r_, c=r_), color='orange')
    plt.xlabel('Step')
    ax1.set_ylabel('Size of the largest connected component')
    ax2.set_ylabel('Probability density')
    plt.title('Logistic regression for the largest connected component over time for thetastar of {}'.format(thetastar))
    plt.show()
