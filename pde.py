import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math
from mpl_toolkits import mplot3d

# Implementation of 1D heat equation using Crank-Nicolson method

# [xl, xr] -> space interval
# [yb, yt] -> time interval
# M -> steps in space direction
# N -> steps in time direction
# D -> diffusion coefficient
# f -> function

# Awj = Bwj-1 + sigma(sj-1 + sj)
def heat_solver(xl, xr, yb, yt, M, N, D, f, lbc, rbc):
    sol = np.zeros((M-1, M-1))
    dx = (xr-xl)/M
    dt = (yt-yb)/N
    sig = D*dt/np.power(dx, 2)
    m = M-1
    n = N
    A = 2 * np.diag(np.ones(m)) + 2 * sig * np.diag(np.ones(m)) + sig * np.diag(np.ones(m-1), 1) + sig * np.diag(np.ones(m-1), -1) # matrix A
    print(A)
    B = 2 * np.diag(np.ones(m)) - 2 * sig * np.diag(np.ones(m)) + sig * np.diag(np.ones(m-1), 1) + sig * np.diag(np.ones(m-1), -1) # matrix B
    print(B)
    lside = np.zeros((n, 1))
    rside = np.zeros((n, 1))
    for i in range(0, n):
        lside[i] = lbc(yb + i * dt)
        rside[i] = rbc(yb + i * dt)
    c1 = np.zeros(m)    # initial condition at t = 0
    for i in range(0, m):
        c1[i] = f(xl + i * dx)
    sol[m-1, :] = c1  # first row of solution = IC
    print(sol)
    w = c1;
    # iterating through time (keep updating w to solve for more rows of the mesh)
    for j in range(1, n):
        sb =  lbc(yb + (j-1) * dt)   # boundary conditions
        st =  rbc(yt + (j-1) * dt)   # boundary conditions
        sj1 = np.zeros(m)
        sj1[0] = sb
        sj1[m-1] = st
        sb =  lbc(yb + j * dt)   # boundary conditions
        st =  rbc(yt + j * dt)   # boundary conditions
        sj = np.zeros(m)
        sj[0] = sb
        sj[m-1] = st
        sj = sig * (sj + sj1)
        # sides = np.zeros((m, 1))
        # sides[0, :] = lside[j] + lside[j+1]
        # sides[m-1, :] = rside[j] + rside[j+1]
        # print("B x sol col: ", np.matmul(B, np.transpose(sol[:, j])))
        # print("sides: ", sig * sides)
        # print("sides: ", sides.shape)
        # R1 = np.matmul(sol[:, j], B)
        # for i in range(0, m):
        #     R2 = np.zeros((m, 1))
        #     R2[i][0] = R1[i]
        R = np.matmul(B, w) + sj
        # print("R: ", R)
        # print("Shape of R:", R.shape)
        # print("Shape of A:", A.shape)
        # print("Shape of sol:", sol.shape)
        #sol[:, j+1] = np.linalg.tensorsolve(A, R)
        x = np.linalg.tensorsolve(A, R)
        w = x
        sol[m-1-j] = w
    # sol = np.concatenate(lside, sol, rside)
    return sol
    # np.innerproduct <B^T y, Ax>


def f(x):
    # return 10
    return math.sin(2 * math.pi * x) * math.sin(2 * math.pi * x)

def p(x):
    return 0;

def crank_nicolson(xl, xr, yb, yt, M, N, f, l, r):
    D = 1
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
    print("lside: ", lside)
    w = np.zeros((m, n+1))
    for i in range(0, m):
        w[i][0] = f(xl + (i+1)*h)
    print("w: ", w)
    for j in range(1, n+1):
        sides = np.zeros(m)
        print("lside: ", lside[j-1] + lside[j])
        print("rside: ", rside[j-1] + rside[j])
        sides[0] = lside[j-1] + lside[j]
        sides[m-1] = rside[j-1] + rside[j]
        print("sides: ", sides)
        #print("j: ", j)
        #print("b: ", b)
        #print("(w[:, j-1]): ", (w[:, j-1]))
        #print("result ", np.matmul(b, (w[:, j-1])))
        print("bwj: ", np.matmul(b, (w[:, j-1])))
        print("A:", a)
        r = np.matmul(b, (w[:, j-1])) + sigma*sides
        w[:, j] = np.linalg.solve(a, r)
        print("w[:, j]: ", w[:, j])
        print("w: ", w)
    w = np.vstack((w, rside))
    w = np.vstack((lside, w))
    return w;


sol = crank_nicolson(0, 1, 0, 1, 100, 100, f, p, p)
print(sol)
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
t, x = np.meshgrid(t, x)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(x, t, np.transpose(sol), 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
#plt.show()

sns.heatmap(sol)
plt.show()
