
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

max_x = 5
max_rand = 10

x = np.arange(0, max_x)
ax.set_ylim(0, max_rand)
line, = ax.plot(x, np.random.randint(0, max_rand, max_x))

def init():  # give a clean slate to start
    line.set_ydata([np.nan] * len(x))
    return line,

def animate(i):  # update the y values (every 1000ms)
    line.set_ydata(np.random.randint(0, max_rand, max_x))
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1000, blit=True, save_count=10)

plt.show()
