import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def laplacian(uk):
    left = np.concatenate((uk[:,-1][:,None], uk[:,:-1]), axis=1)
    right = np.concatenate((uk[:,1:], uk[0,:][:,None]), axis=1)
    top = np.concatenate((uk[-1,:][None,:], uk[:-1,:]), axis=0)
    bottom = np.concatenate((uk[1:,:], uk[0,:][None,:]), axis=0)
    lap = left + right + top + bottom - 4 * uk

    return lap

def calc(u):
    for k in range(0, max_iter_time):
        u[k+1]= u[k] + gamma * (
            laplacian((-u[k] + u[k]**3 - laplacian(u[k])))
        )
    return u

def animate(k):
    if k % 1000 == 0:
        plotheatmap(u[k], k)

if __name__ == "main":

    # initialize mesh, should get the mesh and solver parameters from th
    plate_length = 100
    T = 100
    alpha = 1
    delta_x = 1
    delta_t = 0.01
    max_iter_time = int(T/delta_t)
    Ngridx = int(plate_length/delta_x)
    gamma = (alpha * delta_t) / (delta_x ** 2)

    u = np.empty((1+max_iter_time, Ngridx, Ngridx), dtype="float64")
    u[0] = np.random.uniform(-0.2, 0.2, (Ngridx, Ngridx))

    t0 = time.time()
    u = calc(u)
    t1 = time.time()
    print('running solver for time = %.3f' %(t1-t0))
    plt.figure()
    plt.pcolormesh(u[-1], cmap=plt.cm.viridis, vmin=-1, vmax=1)
    plt.savefig('/Users/fanc/git/amr_524/sp_final.png')
