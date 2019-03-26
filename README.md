# b-e-e-s

# 1. Getcha dependencies in line
After cloning, pip install - r requirements.txt 

# 2. Run the simulation
python bees_no_food.py <U or R> <Y or N>, where U = uniform initial distribution, R = random initial distribution, Y = write data to local .json files, N = plot data directly from within the program

The program will prompt you for parameters such as agent count, step count, side length, and whether you wish to measure the density or the theta* variable. I've always been running it with theta*, so the density option may be bugged at this time.
  
# 3. If you elected to write data...
python plot_random_walk_results.py 

This program will also prompt for parameters, which should match those provided to bees_no_food.py.
