import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import pandas as pd
import time
import mpl_toolkits.mplot3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Implementation of 1D heat equation using Crank-Nicolson method

# [xl, xr] -> space interval
# [yb, yt] -> time interval
# M -> steps in space direction
# N -> steps in time direction
# D -> diffusion coefficient
# f -> function
# l -> boundary condition
# r -> boundary condition
# Awj = Bwj-1 + sigma(sj-1 + sj)
def crank_nicolson(xl, xr, yb, yt, M, N, D, f, l, r):
    h = (xr-xl)/M
    k = (yt-yb)/N
    sigma = D*k/(h*h)
    m = M-1
    n = N
    a = 2 * np.diag(np.ones(m)) + 2 * sigma * np.diag(np.ones(m)) + (-1 * sigma) * np.diag(np.ones(m-1), 1) + (-1 * sigma) * np.diag(np.ones(m-1), -1)
    b = 2 * np.diag(np.ones(m)) - 2 * sigma * np.diag(np.ones(m)) + sigma * np.diag(np.ones(m-1), 1) + sigma * np.diag(np.ones(m-1), -1)
    lside = np.zeros(n+1, )
    rside = np.zeros(n+1, )
    for i in range(0, n+1):
        lside[i] = l(yb + i*k)
        rside[i] = r(yb + i*k)
    w = np.zeros((m, n+1))
    for i in range(0, m):
        w[i][0] = f(xl + (i+1)*h)
    for j in range(1, n+1):
        sides = np.zeros(m)
        sides[0] = lside[j-1] + lside[j]
        sides[m-1] = rside[j-1] + rside[j]
        r = np.matmul(b, (w[:, j-1])) + sigma*sides
        w[:, j] = np.linalg.solve(a, r)
        #temp = np.vstack((w, rside))
        #temp = np.vstack((lside, temp))
        #sns.heatmap(temp)
        #plt.show()
    w = np.vstack((w, rside))
    w = np.vstack((lside, w))
    return w;

def crank_nicolson_anim(xl, xr, yb, yt, M, N, D, f, l, r, zarray):
    h = (xr-xl)/M
    k = (yt-yb)/N
    sigma = D*k/(h*h)
    m = M-1
    n = N
    a = 2 * np.diag(np.ones(m)) + 2 * sigma * np.diag(np.ones(m)) + (-1 * sigma) * np.diag(np.ones(m-1), 1) + (-1 * sigma) * np.diag(np.ones(m-1), -1)
    b = 2 * np.diag(np.ones(m)) - 2 * sigma * np.diag(np.ones(m)) + sigma * np.diag(np.ones(m-1), 1) + sigma * np.diag(np.ones(m-1), -1)
    lside = np.zeros(n+1, )
    rside = np.zeros(n+1, )
    for i in range(0, n+1):
        lside[i] = l(yb + i*k)
        rside[i] = r(yb + i*k)
    w = np.zeros((m, n+1))
    for i in range(0, m):
        w[i][0] = f(xl + (i+1)*h)
    for j in range(1, n+1):
        sides = np.zeros(m)
        sides[0] = lside[j-1] + lside[j]
        sides[m-1] = rside[j-1] + rside[j]
        r = np.matmul(b, (w[:, j-1])) + sigma*sides
        w[:, j] = np.linalg.solve(a, r)
        temp = np.vstack((w, rside))
        temp = np.vstack((lside, temp))
        zarray[:,:,j] = temp
    return zarray;

def f(x):
    #return 10
    #return math.sin(2 * math.pi * x) * math.sin(2 * math.pi * x)
    return np.exp(-0.5 * x)


def p(x):
    #return 0;
    #return 10;
    return np.exp(x)

def q(x):
    #return 0;
    #return 10;
    return np.exp(x - 0.5)

def do_anim(xl, xr, yb, yt, M, N, D, f, l, r):
    frn = M+1 # frame number of animation
    fps = 10 # frame per sec
    x = np.linspace(-1, 1, M+1)
    t = np.linspace(-1, 1, N+1)
    t, x = np.meshgrid(t, x)
    zarray = np.zeros((M+1, N+1, frn))

    zarray = crank_nicolson_anim(xl, xr, yb, yt, M, N, D, f, l, r, zarray)
    max = np.amax(zarray)
    min = np.amin(zarray)
    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, t, zarray[:,:,frame_number], cmap="magma")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(x, t, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(min - 0.2 * min, max + 0.2 * max)
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)

    fn = 'plot_surface_animation_funcanimation'
    ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    plt.show()

def do_static(xl, xr, yb, yt, M, N, D, f, l, r):
    x = np.linspace(-1, 1, M+1)
    t = np.linspace(-1, 1, N+1)
    t, x = np.meshgrid(t, x)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    sol = crank_nicolson(xl, xr, yb, yt, M, N, D, f, l, r)
    ax.plot_surface(x, t, sol, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('temp');
    plt.show()

#do_anim(0, 1, 0, 1, 100, 100, 1, f, p, q)
do_static(0, 1, 0, 1, 100, 100, 1, f, p, q)
