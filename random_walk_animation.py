import numpy as np
import math
import matplotlib.pyplot as plt
import Bee
import simulation_helpers
import random
from matplotlib.animation import FuncAnimation
import itertools


plt.style.use('seaborn-pastel')


def animate_walk(num_bees, side_length, step_count):
    bee_array = []
    total_bees = num_bees
    thetastar = [np.linspace(-(math.pi / 2), (math.pi / 2), 100)]
    thetastar = list(thetastar[0])
    n = side_length
    steps = step_count
    tstar_range = 100
    food_donation_percent = 0.50
    food_transition_rate = 1

    for i in range(0, total_bees):
        bee_array.append(Bee.Bee(i, total_bees, thetastar, tstar_range, 0.1, n, steps, "uniform", True))

    for step in range(1, steps):
        for bee in bee_array:
            bee.move(step)
        for bee_1, bee_2 in itertools.combinations(bee_array, 2):
            simulation_helpers.setup_trophallaxis(step, bee_1, bee_2, food_donation_percent, food_transition_rate)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, n), ylim=(0, n))
    xdatas = {n: [] for n in range(0, total_bees)}
    ydatas = {n: [] for n in range(0, total_bees)}

    lines = [ax.plot([], [], '*')[0] for _ in bee_array]
    r_set = set(np.linspace(0, 1, num=total_bees))
    g_set = set(np.linspace(0, 1, num=total_bees))
    b_set = set(np.linspace(0, 1, num=total_bees))
    for line in lines:
        r = random.sample(r_set, 1)[0]
        g = random.sample(g_set, 1)[0]
        b = random.sample(b_set, 1)[0]
        line.set_color((r, g, b))

    def animate(i, bees, lines):
        for line, bee in zip(lines, bees):
            xdatas[bee.number].append(bee.trace.get(i)[0])
            ydatas[bee.number].append(bee.trace.get(i)[1])
            line.set_data(xdatas[bee.number][0], ydatas[bee.number][0])
            xdatas[bee.number].pop(0)
            ydatas[bee.number].pop(0)
        return lines

    ax.set_xlim([0.0, n])
    ax.set_xlabel('X')

    ax.set_ylim([0.0, n])
    ax.set_ylabel('Y')

    ax.set_title('2D Walk Test')

    anim = FuncAnimation(fig, animate, frames=steps, fargs=(bee_array, lines),
                         interval=1000, blit=False)

    # anim.save('bee_paths.gif', writer='pillow')
    plt.show()


animate_walk(16, 25, 100)
