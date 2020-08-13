import numpy as np
import Simulation
import math
import simulation_helpers

thetastars = [np.linspace(-(math.pi / 2), (math.pi / 2), simulation_helpers.TSTAR_RANGE)]
simulation = Simulation.Simulation(num_bees=16, side_length=25, step_count=100, thetastars=thetastars)
simulation.run()
simulation.animate_walk()
