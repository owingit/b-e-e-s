import numpy as np
import matplotlib.pyplot as plt
import simulation
import random
import math
import Bee
import simulation_helpers
from matplotlib.animation import FuncAnimation


plt.style.use('seaborn-pastel')


def animate_walk(num_bees, side_length, step_count, thetastars):
    total_bees = num_bees
    n = side_length
    steps = step_count
    bee_array = []
    thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
    for i in range(0, total_bees):
        bee_array.append(Bee.Bee(i, total=total_bees, tstar=thetastar, tstar_range=simulation_helpers.TSTAR_RANGE,
                                 initially_fed_percentage=0.1, n=n, steps=steps, r_or_u="uniform",
                                 use_periodic_boundary_conditions=True))
    simulation.simulate(bee_array, step_count)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, n), ylim=(0, n))
    xdatas = {n: [] for n in range(0, total_bees)}
    ydatas = {n: [] for n in range(0, total_bees)}

    bee_paths = [ax.plot([], [], '*')[0] for _ in bee_array]
    r_set = set(np.linspace(0, 1, num=total_bees))
    g_set = set(np.linspace(0, 1, num=total_bees))
    b_set = set(np.linspace(0, 1, num=total_bees))
    for line in bee_paths:
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

    anim = FuncAnimation(fig, animate, frames=steps, fargs=(bee_array, bee_paths),
                         interval=1000, blit=False)

    # anim.save('bee_paths.gif', writer='pillow')
    plt.show()


thetastars = [np.linspace(-(math.pi / 2), (math.pi / 2), simulation_helpers.TSTAR_RANGE)]
animate_walk(16, 25, 100, thetastars)
