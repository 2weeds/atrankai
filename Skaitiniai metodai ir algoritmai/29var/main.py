import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PyFunkcijos import *

CONST_alpha = np.array([1, 1, 1, 1])
CONST_max_iter = 1000
CONST_eps = 1e-12
e = 1e-12
A1 = np.matrix([
    [2, 5, 1, 2],
    [-2, 0, 3, 5],
    [1, 0, -1, 1],
    [0, 5, 4, 7]]).astype(np.float64)
b1 = np.matrix([[-1], [7], [3], [4]]).astype(np.float64)

A2 = np.matrix([
    [4, 3, -1, 1],
    [3, 9, -2, -2],
    [-1, -2, 11, -1],
    [1, -2, -1, 5]]).astype(np.float64)
b2 = np.matrix([[12], [10], [-28], [16]]).astype(np.float64)


# %% first task
def GaussSeidel(matrix_A, matrix_b, alpha, max_epochs, eps):
    if np.linalg.det(matrix_A) == 0:
        return -1
    n = np.shape(matrix_A)[0]
    Atld = np.diag(1. / np.diag(matrix_A)).dot(matrix_A) - np.diag(alpha)
    btld = np.diag(1. / np.diag(matrix_A)).dot(matrix_b)
    x = np.zeros(shape=(n, 1))
    x1 = np.zeros(shape=(n, 1))
    prec = []
    for it in range(0, max_epochs):
        for i in range(0, n):
            x1[i] = (btld[i] - Atld[i, :].dot(x1)) / alpha[i]
            prec.append((np.linalg.norm(x1 - x) / (np.linalg.norm(x) + np.linalg.norm(x1))))
        if prec[it] < eps:
            break
        x[:] = x1[:]
    return prec, x


def checkResults(matrix_A, x):
    for i in range(0, 4):
        tmp = 0
        for j in range(0, 4):
            tmp += matrix_A[i, j] * x[j, 0]
        print(tmp)


_, x = GaussSeidel(A2, b2, CONST_alpha, CONST_max_iter, CONST_eps)
checkResults(A2, x)


def plot(y, alpha, equation):
    x = np.arange(len(y))
    plt.plot(x, y, label='$alpha = {i}$'.format(i=alpha))
    plt.ylim(1e-12, 2e-1)
    plt.yscale('logit')
    plt.title(equation)


def ExecuteFirstTask():
    for i in range(0, 4):
        CONST_alpha[:] = i + 1
        prec = GaussSeidel(A1, b1, CONST_alpha, CONST_max_iter, CONST_eps)
        if prec == -1:
            print("Matrix is singular")
            break
        plot(prec, CONST_alpha, "First equation")
    plt.legend(loc='best')
    plt.show()
    for i in range(0, 4):
        CONST_alpha[:] = i + 1
        prec, _ = GaussSeidel(A2, b2, CONST_alpha, CONST_max_iter, CONST_eps)
        plot(prec, CONST_alpha, "Second equation")
    plt.legend(loc='best')
    plt.show()


def QR_decomposition(A, b):
    n = 4
    Q = np.identity(n)
    for i in range(0, n - 1):
        z = A[i:n, i]
        zp = np.zeros(np.shape(z))
        zp[0] = np.linalg.norm(z)
        omega = z - zp
        omega = omega / np.linalg.norm(omega)
        Qi = np.identity(n - i) - 2 * omega * omega.transpose()
        A[i:n, :] = Qi.dot(A[i:n, :])
        Q = Q.dot(
            np.vstack(
                (
                    np.hstack((np.identity(i), np.zeros(shape=(i, n - i)))),
                    np.hstack((np.zeros(shape=(n - i, i)), Qi))
                )
            )
        )
    # atgalinis etapas:
    b1 = Q.transpose().dot(b)
    x = np.zeros(shape=(n, n))
    for i in range(n - 1, -1, -1):
        x[i, :] = (b1[i, :] - A[i, i + 1:n] * x[i + 1:n, :]) / A[i, i]
    return x


# %% second task
# Netiesines lygciu sistemos

def LF(x):  # grazina reiksmiu stulpeli
    s = np.matrix([[-(5*x[1])/(x[0]**2+1)+x[1]**2-x[0]**2], [x[0]**2+x[1]**2-12]])
    return s
def LFg(x):  # grazina reiksmiu stulpeli
    s = [-(5*x[1])/(x[0]**2+1)+x[1]**2-x[0]**2, x[0]**2+x[1]**2-12]
    s = np.vstack(s)
    return s


def f(x):
    f1 = -(5*x[1])/(x[0]**2+1)+x[1]**2-x[0]**2
    f2 = x[0]**2+x[1]**2-12
    return [f1, f2]


def function(xy):
    x, y = xy
    return [-(5*y)/(x**2+1)+y**2-x**2,
            x**2+y**2-12]


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
    plt.contour(x, y, -(5*y)/(x**2+1)+y**2-x**2, [0], colors="b")
    plt.contour(x, y, x**2+y**2-12, [0], colors="g")
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

# Broideno metodas
def Broiden(funcCount, x1, func):

    dx = np.sum(np.abs(x1)) * e
    A = np.matrix(np.zeros((funcCount, funcCount)))
    x2 = np.zeros((funcCount, 1))
    A, x2 = JacobianStart(A, funcCount, x1, dx, func)
    f1 = func(x1)
    iter = 0
    for i in range(100):
        iter += 1
        s = -np.linalg.solve(A, f1)
        x2 = np.matrix(x1 + s)
        f2 = func(x2)
        y = f2 - f1
        A += (y - A * s) * s.transpose() / (s.transpose() * s)
        tikslumas = np.linalg.norm(s) / (np.linalg.norm(x1) + np.linalg.norm(s))
        print('iteracija: = {0}'.format(iter))
        print('x reiksme: = {0}'.format(x2))
        print('tikslumas: = {0}'.format(tikslumas))
        f1 = f2
        x1 = x2
        if tikslumas < e:
            break
        if iter == 100 and tikslumas > e:
            print('Tikslumas nepasiektas!')
    print('iteracija: = {0}'.format(iter))
    print('x reiksme: = \n {0}'.format(x1))
    return x1

def JacobianStart(A, n, x1, dx, func):
    for i in range(n):
        x2 = np.matrix(x1)
        x2[i] += dx
        A[:, i] = (func(x2) - func(x1)) / dx
    return A, x2


def BroidenGraph(x1, x2):
    fig = plt.figure(1, figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylabel('z')
    plt.draw()
    xSpace = np.linspace(-6, 6, 20)
    ySpace = np.linspace(-6, 6, 20)
    xGrid, yGrid = np.meshgrid(xSpace, ySpace)
    z = Surface(xSpace, ySpace, xGrid, yGrid)
    ax1.contour(xGrid, yGrid, z[:, :, 0], [0], colors='b')
    ax1.contour(xGrid, yGrid, z[:, :, 1], [0], colors='g')
    ax1.plot(x1[0], x1[1], markersize=10, color='black', marker='o')
    ax1.plot(x2[0], x2[1], markersize=10, color='red', marker='o')
    plt.show()

def Surface(xSpace, ySpace, xGrid, yGrid):
    z = np.zeros((len(xSpace), len(ySpace), 2))
    for i in range(len(xSpace)):
        for j in range(len(ySpace)):
            z[i, j, :] = LF([xGrid[i][j], yGrid[i][j]]).transpose()
    return z

def Graph():
    fig = plt.figure(1, figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylabel('z')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_ylabel('z')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_ylabel('z')
    plt.draw()
    xSpace = np.linspace(-6, 6, 20)
    ySpace = np.linspace(-6, 6, 20)
    xGrid, yGrid = np.meshgrid(xSpace, ySpace)
    z = Surface(xSpace, ySpace, xGrid, yGrid)
    ax1.plot_surface(xGrid, yGrid, z[:, :, 0], color='blue', alpha=0.4)
    ax1.contour(xGrid, yGrid, z[:, :, 0], [0], colors='b')
    ax2.plot_surface(xGrid, yGrid, z[:, :, 1], color='purple', alpha=0.4)
    ax2.contour(xGrid, yGrid, z[:, :, 1], [0], colors='g')
    ax3.contour(xGrid, yGrid, z[:, :, 0], [0], colors='b')
    ax3.contour(xGrid, yGrid, z[:, :, 1], [0], colors='g')
    plt.show()


def main():
    Graph()
    # ExecuteFirstTask()
    # x = QR_decomposition(A2, b2)
    # print(x[:, 0])
    plotNonLinearEquations()
    print('LF(x) skaiciavimas')
    funcCount = 2
    x1 = np.matrix(np.zeros((funcCount, 1)))
    # Pradinis taskas
    x1 = [-3.0, 2]
    x1 = np.vstack(x1)
    x2 = Broiden(funcCount, x1, LFg)
    BroidenGraph(x1, x2)

main()
