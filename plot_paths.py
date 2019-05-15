import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib

num_agents = int(input("How many agents were simulated?"))
num_steps = int(input("How many steps were simulated?"))
side_length = int(input("What were the box dimensions?"))
filename = raw_input("Please provide a file and we'll plot the agent paths. ")
thetastar = filename[:5]
all_paths = np.load(filename)
print all_paths
fig = plt.figure()
ax = fig.add_subplot(111)


def break_side_wrap(values):
    values1, values2 = itertools.tee(values)

    # Yield the first value
    yield next(values2)

    for (prev_x, prev_y), (x, y) in itertools.izip(values1, values2):

        # If the data wraps over the top
        if y > prev_y and y - prev_y > side_length/2:
            yield (x, y - side_length)
            yield (x, None)
            yield (prev_x, prev_y + side_length)

        # If the data wraps under the bottom
        elif y < prev_y and prev_y - y > side_length/2:
            yield (x, y + side_length)
            yield (x, None)
            yield (prev_x, prev_y - side_length)

        # Add each original value
        yield (x, y)

colors = []
for name, _ in matplotlib.colors.cnames.iteritems():
	if 'white' not in name:
		colors.append(name)
print len(colors)

#for bee in range(0, num_agents):
#	for i in range(0, num_steps):
#		ax.scatter(i, all_paths[bee][i][0], all_paths[bee][i][1], alpha=0.3, color=colors[bee])

_of_xs = []
_of_ys = []
_of_clrs = []
for bee in range(0, num_agents):
	xs = []
	ys = []
	clrs = []
	for i in range(0, num_steps):
		xs.append(all_paths[bee][i][0])
		ys.append(all_paths[bee][i][1])
		clrs.append(all_paths[bee][i][2])
	_of_xs.append(xs)
	_of_ys.append(ys)
	_of_clrs.append(clrs)

list_of_xs = np.array(_of_xs)
list_of_ys = np.array(_of_ys)
list_of_clrs = np.array(_of_clrs)
concat = np.concatenate(list_of_clrs)

for i in range(0, num_agents):
	plot_xs = list_of_xs[i]
	plot_ys = list_of_ys[i]
	plot_clrs = list_of_clrs[i]
	print plot_xs, plot_ys, plot_clrs
	norms = [float(j)/max(plot_clrs) for j in plot_clrs]
	
	ax.scatter(plot_xs, plot_ys, alpha=0.5, c=norms, cmap="viridis", label=norms)
	ax.scatter(plot_xs[0], plot_ys[0], alpha=1, color="red", marker='x')
	ax.scatter(plot_xs[len(plot_xs)-1], plot_ys[len(plot_ys)-1], alpha=1, color="red", marker='*')
	# ax.plot(plot_xs[i], plot_ys[i], alpha=0.1, c=plot_clrs, cmap="bwr_r")
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.legend()
ax.set_title('Paths of {} agents taking {} steps through a {}x{} arena with theta* of {}'.format(num_agents, num_steps, side_length, side_length, thetastar))
plt.tight_layout(pad=0)

plt.show()
