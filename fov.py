import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def arc_patch(center, theta1, theta2, ax=None, resolution=50, **kwargs):
    # fig, ax = plt.subplots(1,2)
    # center =[0,0]
    # theta1 = -30
    # theta2 = 30
    # radius = 1
    # resolution = 50
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    radius = 0.3
    points = np.vstack((radius*np.cos(theta) + center[0],
                        radius*np.sin(theta) + center[1]))
    radius = 3.5
    theta = theta[::-1]
    point_outer = np.vstack((radius*np.cos(theta) + center[0],
                        radius*np.sin(theta) + center[1]))
    points = np.concatenate((points, point_outer), axis=1)
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True, alpha=0.2, **kwargs)
    ax.add_patch(poly)
    return poly


fig, ax = plt.subplots(1,2)

# @jeanrjc solution, which might hide other objects in your plot
# ax[0].plot([-1,1],[1,-1], 'r', zorder = -10)
# filled_arc((0.,0.3), 1, 90, 180, ax[0], 'blue')
# ax[0].set_title('version 1')

# simpler approach, which really is just the arc
ax[1].plot([-1,1],[1,-1], 'r', zorder = -10)
arc_patch((0.,0.3), -30, 30, ax=ax[1], fill=True, color='blue')
ax[1].set_title('version 2')

# axis settings
for a in ax:
    a.set_aspect('equal')
    a.set_xlim(-0.5, 6)
    a.set_ylim(-3, 3)

plt.show()
