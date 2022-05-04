import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def show_animation(t_all, pos_all):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    routine, = ax.plot([], [], '--', lw=1, color='blue')
    title_time = ax.text(0.05, 0.95, "", transform=ax.transAxes)
    x = pos_all[:,0]
    y = pos_all[:,1]

    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    ax.set_xlim([mid_x-maxRange, mid_x+maxRange])
    ax.set_ylim([mid_y+maxRange, mid_y-maxRange])

    def update_routine(i):
        time = t_all[i]
        x_from0 = pos_all[0:i+1, 0]
        y_from0 = pos_all[0:i+1, 1]
        print(time,x_from0,y_from0)
        print(type(routine))
        routine.set_data(x_from0, y_from0)
        routine.set_markersize(20)
        title_time.set_text(u"Time = {:.2f} s".format(time))
        return routine



    # Creating the Animation object
    routine_ani = animation.FuncAnimation(fig, update_routine, interval=1000)
    plt.show()
    return routine_ani


if __name__ == "__main__":
    ani = show_animation([1, 2, 3,4,5,6,7,8,9], np.array([[40, 10], [50, 65], [63, 70], [40, 5], [50, 60], [60, 35],[40, 5], [50, 60], [25, 70],[40, 5], [50, 60], [60, 70]]))
