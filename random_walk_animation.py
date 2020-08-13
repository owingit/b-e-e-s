import numpy as np
import matplotlib.pyplot as plt
import Simulation
import random
import math
import Bee
import simulation_helpers
from matplotlib.animation import FuncAnimation


plt.style.use('seaborn-pastel')


def animate_walk(num_bees, side_length, step_count, thetastars):
    simulation = Simulation.Simulation(num_bees, side_length, step_count, thetastars)
    simulation.simulate()

    fig = plt.figure()
    ax = plt.axes(xlim=(0, simulation.n), ylim=(0, simulation.n))
    xdatas = {n: [] for n in range(0, simulation.total_bees)}
    ydatas = {n: [] for n in range(0, simulation.total_bees)}

    bee_paths = [ax.plot([], [], '*')[0] for _ in simulation.bee_array]
    r_set = set(np.linspace(0, 1, num=simulation.total_bees))
    g_set = set(np.linspace(0, 1, num=simulation.total_bees))
    b_set = set(np.linspace(0, 1, num=simulation.total_bees))
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

    ax.set_xlim([0.0, simulation.n])
    ax.set_xlabel('X')

    ax.set_ylim([0.0, simulation.n])
    ax.set_ylabel('Y')

    ax.set_title('2D Walk Test')

    anim = FuncAnimation(fig, animate, frames=simulation.steps, fargs=(simulation.bee_array, bee_paths),
                         interval=1000, blit=False)

    # anim.save('bee_paths.gif', writer='pillow')
    plt.show()


thetastars = [np.linspace(-(math.pi / 2), (math.pi / 2), simulation_helpers.TSTAR_RANGE)]
animate_walk(16, 25, 100, thetastars)
