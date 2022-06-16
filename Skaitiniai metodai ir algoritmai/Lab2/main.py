# %% imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols,diff,Symbol
from matplotlib import cm

from PyFunkcijos import *
import math
from numpy import arange, meshgrid, sqrt
from scipy.optimize import fsolve
from sympy.utilities.lambdify import lambdify
from sympy.solvers import solve

# %%    marticų koeficientai
Singular_A = np.matrix([
    [1, -3],
    [0, 0]]).astype(np.float64)

A1 = np.matrix([
    [3, 1, -1, 5],
    [-3, 4, -8, -1],
    [1, -3, 7, 6],
    [0, 5, -9, -4]
]).astype(np.float64)
A2 = np.matrix([
    [5, 2, 0, 1],
    [2, -5, 0, 2],
    [9, -6, -6, 1],
    [1, 2, 1, 1]
]).astype(np.float64)
# %%    b vektoriai
b1 = (np.matrix([20, -36, 41, -16])).transpose().astype(np.float64)
b2 = (np.matrix([19, -5, 39, 5])).transpose().astype(np.float64)
b11 = np.array([[20], [-36], [41], [-16]]).astype(np.float64)
b22 = np.array([[19], [-5], [39], [5]]).astype(np.float64)
Singular_b = (np.matrix([5, 0])).transpose().astype(np.float64)  # laisvuju nariu vektorius-stulpelis
# %%    konstantos
CONST_e = 1e-12  # tikslumo konstanta
CONST_alpha = np.array([0, 0, 0, 0])  # laisvai parinkti metodo parametrai
CONST_max_epochs = 1000


# %%    pirmoji užduotis

def GaussMethod(matrix_A, matrix_b):
    n = (np.shape(matrix_A))[0]  # lygciu skaicius nustatomas pagal ivesta matrica A
    nb = (np.shape(matrix_b))[1]  # laisvuju nariu vektoriu skaicius nustatomas pagal ivesta matrica b
    aug_matrix = np.hstack((matrix_A, matrix_b))  # isplestoji matrica
    print("A = \n", matrix_A[0:4, 0:4])
    print("b = \n", matrix_b[:, 0])

    # tiesioginis etapas:
    for i in range(0, n - 1):  # range pradeda 0 ir baigia n-2 (!)
        for j in range(i + 1, n):  # range pradeda i+1 ir baigia n-1
            aug_matrix[j, i:n + nb] = aug_matrix[j, i:n + nb] - aug_matrix[i, i:n + nb] * aug_matrix[j, i] / aug_matrix[
                i, i]
            aug_matrix[j, i] = 0
        print(i + 1, "iteration")
        print(aug_matrix)
        print()

    #  atvirkstinis etapas:

    if aug_matrix[aug_matrix.shape[0] - 1, aug_matrix.shape[1] - 2] == 0:
        if aug_matrix[aug_matrix.shape[0] - 1, aug_matrix.shape[1] - 1] == 0:
            return print("sprendinių be galo daug")
        else:
            return print("sprendinių nėra")
    x = np.zeros(shape=(n, nb))
    for i in range(n - 1, -1, -1):  # range pradeda n-1 ir baigia 0 (trecias parametras yra zingsnis)
        x[i, :] = (aug_matrix[i, n:n + nb] - aug_matrix[i, i + 1:n] * x[i + 1:n, :]) / aug_matrix[i, i]
    for i in range(len(matrix_b)):
        print("x", i + 1, "=", "{:.2f}".format(x[i, 0]))
    ans = np.zeros(shape=(n, nb))
    for i in range(0, n):
        for j in range(0, n):
            ans[i, 0] = ans[i, 0] + x[j, 0] * matrix_A[i, j]
    ansx = np.zeros(shape=(n, nb))
    # for i in range(0,n):
    #     for j in range(0,n):
    #         ansx[i,0] = ansx[i,0]+ans[i,j]
    ans.transpose()
    print()
    print("Initial matrix b")
    for i in range(len(ans)):
        print("{:.2f}".format(ans[i, 0]))

gA = np.matrix([
    []
])
def SimpleIterationMethod(A, b, alpha, max_epochs, precision):
    if np.linalg.det(A) == 0:
        return print("LinAlgErr: Singular matrix")
    n = np.shape(A)[0]
    # laisvai parinkti metodo parametrai
    Atld = np.diag(1. / np.diag(A)).dot(A) - np.diag(alpha)
    print(np.diag(A).dot(A))
    btld = np.diag(1. / np.diag(A)).dot(b)
    prec = []  # tuscias sarasas tikslumo reiksmiukaupimui
    x = np.zeros(shape=(n, 1))
    x1 = np.zeros(shape=(n, 1))
    for it in range(0, max_epochs):
        x1 = ((btld - Atld.dot(x)).transpose() / alpha).transpose()
        prec.append(
            np.linalg.norm(x1 - x, ord=np.inf) / (np.linalg.norm(x, ord=np.inf) + np.linalg.norm(x1, ord=np.inf))
        )
        if prec[it] < precision:
            break
        x[:] = x1[:]
    print(x.transpose())
    return prec


def plot(y, alpha, equation):
    x = np.arange(len(y))
    plt.plot(x, y, label='$alpha = {i}$'.format(i=alpha))
    plt.ylim(1e-12, 2e-1)
    plt.yscale('logit')
    plt.title(equation)


# %%    antroji užduotis
def LF(x):  # grazina reiksmiu stulpeli
    s = np.matrix([[x[0] ** 2 + 2 * (x[1] - np.cos(x[0])) ** 2 - 20], [x[0] ** 2 * x[1] - 2]])
    return s


def f(x):
    f1 = 0.1*x[0]**2 + 0.2 * x[1]**2-4
    f2 = x[0]+x[1]**2+0.1
    return [f1, f2]


def function(xy):
    x, y = xy
    return [0.1*x**2 + 0.2 * y**2-4,
            x+y**2+0.1]


def jacobian(xy):
    x, y = xy
    return [[0.2*x, 0.4*y],
            [4*x, 1+2*y]]

def gfun(xy):
    x,y = xy
    return [x+y**2-2,
            x**2-y**2-4]
def gfunjacobi(xy):
    x,y = xy
    return [[1,2*y],
            [2*x,-2*y]]

def iterative_newton(fun, x_init, jacobian_fun):
    max_epochs = 50
    epsilon = 1e-12

    x_last = x_init

    for epoch in range(max_epochs):
        J = np.array(jacobian_fun(x_last))
        F = np.array(fun(x_last))

        diff = np.linalg.solve(J, -F)
        x_last = x_last + diff

        # Stop condition:
        if np.linalg.norm(diff) < epsilon:
            print('convergence!, epoch:', epoch)
            break

    else:  # only if the for loop end 'naturally'
        print('not converged')

    return x_last


def plotNonLinearEquations():
    # ----------------------------------
    fig1 = plt.figure(1, figsize=plt.figaspect(0.5))
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    xx = np.linspace(-5, 5, 50)
    yy = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(xx, yy)
    Z = Pavirsius(X, Y, LF)
    ax1.plot_surface(X, Y, Z[:, :, 0], color='blue', alpha=0.4)
    ax1.plot_wireframe(X, Y, Z[:, :, 0], color='black', alpha=1, linewidth=0.3, antialiased=True)
    ax2.plot_surface(X, Y, Z[:, :, 1], color='purple', alpha=0.4)
    ax2.plot_wireframe(X, Y, Z[:, :, 1], color='black', alpha=1, linewidth=0.3, antialiased=True)
    plt.show()

    delta = 0.025
    x, y = meshgrid(arange(-5, 5, delta), arange(-5, 5, delta))
    plt.contour(x, y, x ** 2 + 2 * (y - np.cos(x)) ** 2 - 20, [0], colors="b")
    plt.contour(x, y, x ** 2 * y - 2, [0], colors="g")
    x1 = fsolve(f, [-4, 4])
    x2 = fsolve(f, [-2, 4])
    x3 = fsolve(f, [0.5, 2])
    x4 = fsolve(f, [3, 1])
    plt.plot(x1[0], x1[1], color="red", marker='o', markersize=5)
    plt.plot(x2[0], x2[1], color="red", marker='o', markersize=5)
    plt.plot(x3[0], x3[1], color="red", marker='o', markersize=5)
    plt.plot(x4[0], x4[1], color="red", marker='o', markersize=5)
    plt.show()


def printNonLinearByPython():
    x = fsolve(f, [-5, 4])
    print("x1 = ", x)
    x = fsolve(f, [-2, 4])
    print("x2 = ", x)
    x = fsolve(f, [0.5, 2])
    print("x3 = ", x)
    x = fsolve(f, [3, 0.5])
    print("x4 = ", x)


# %%     trečioji užduotis
CONST_n = 5  # vieno tinklo parduotuvės mieste kiekis
CONST_m = 5  # planuojamu parduotuviu statymas kiekis
CONST_citySize = 20


def GenerateDots(n):
    x_data = []
    y_data = []
    for i in range(n):
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        while x_data.__contains__(x):
            x = random.randint(-10, 10)
        x_data.append(x)
        while y_data.__contains__(y):
            y = random.randint(-10, 10)
        y_data.append(y)
    return np.array(x_data), np.array(y_data)


def Cx(x1, y1, x2, y2):
    return exp(-0.2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2))


def CRx(x1, y1, xr, yr):
    if -10 <= xr <= 10 and -10 <= yr <= 10:
        return 0
    else:
        return exp(0.25 * ((x1 - xr) ** 2 + (y1 - yr) ** 2)) - 1


def fx0(x0_array, y0_array, x10, y10, xr0, yr0):
    F0 = 0
    if x10 > 10 or x10 < -10 or y10 > 10 or y10 < -10:
        if x10 > 10 and y10 > 10:
            xr0 = 10
            yr0 = 10
        elif x10 < -10 and y10 > 10:
            xr0 = -10
            yr0 = 10
        elif x10 > 10 and y10 < -10:
            xr0 = -10
            yr0 = 10
        elif x10 < -10 and y10 < -10:
            xr0 = -10
            yr0 = 10
        elif -10 < x10 < 10 and y10 < -10:
            xr0 = x10
            yr0 = -10
        elif -10 < x10 < 10 and y10 > 10:
            xr0 = x10
            yr0 = 10
        elif -10 < y10 < 10 and x10 > 10:
            yr0 = y10
            xr0 = 10
        elif -10 < y10 < 10 and x10 < -10:
            yr0 = y10
            xr0 = -10
        for i in range(len(x0_array)):
            F0 += Cx(x0_array[i], y0_array[i], x10, y10)
        F0 += CRx(x10, y10, xr0, yr0)
        return F0
    else:
        for i in range(len(x0_array)):
            F0 += Cx(x0_array[i], y0_array[i], x10, y10)
        return F0


# %%    programos startup
def ExecuteFirstTask():
    print("-" * 65)
    print("Gauss Method")
    print("-" * 65)
    print("First linear equation:")
    print("-" * 65)
    # GaussMethod(Singular_A, Singular_b)
    GaussMethod(A1, b1)
    print("-" * 65)
    print("Second linear equation:")
    print("-" * 65)
    GaussMethod(A2, b2)
    print("-" * 65)
    print("Simple Iteration Method")
    print("-" * 65)
    print("First linear equation:")
    print("-" * 65)
    for i in range(2, 7):
        CONST_alpha[:] = i
        prec = SimpleIterationMethod(A1, b11, CONST_alpha, CONST_max_epochs, CONST_e)
        plot(prec, i, "First linear equation")
    plt.legend(loc='best')
    plt.show()
    print("-" * 65)
    print("Second linear equation:")
    print("-" * 65)
    for i in range(2, 7):
        CONST_alpha[:] = i
        prec = SimpleIterationMethod(A2, b22, CONST_alpha, 1000, CONST_e)
        plot(prec, i, "Second linear equation")
    plt.legend(loc='best')
    plt.show()
    print("-" * 65)
    print("Python numpy solver")
    print("-" * 65)
    print("First linear equation:")
    print("-" * 65)
    print(np.linalg.solve(A1, b11))
    print("-" * 65)
    print("Second linear equation:")
    print("-" * 65)
    print(np.linalg.solve(A2, b22))
    print("-" * 65)
    print("END")
    print("-" * 65)


def ExecuteSecondTask():
    print("Solutions by python")
    printNonLinearByPython()
    plotNonLinearEquations()
    x1 = iterative_newton(function, [2, 6], jacobian)
    print('solution found at:', x1)
    print('F(x1)', function(x1))
    x2 = iterative_newton(function, [-2, 4], jacobian)
    print('solution found at:', x2)
    print('F(x2)', function(x2))
    x3 = iterative_newton(function, [0.5, 2], jacobian)
    print('solution found at:', x3)
    print('F(x3)', function(x3))
    x4 = iterative_newton(function, [3, 0.5], jacobian)
    print('solution found at:', x4)
    print('F(x4)', function(x4))


def ExecuteThirdTask():
    x, y = GenerateDots(CONST_n)
    print(x)
    print(y)


# %% main
# ExecuteFirstTask()
ExecuteSecondTask()
# ExecuteThirdTask()
