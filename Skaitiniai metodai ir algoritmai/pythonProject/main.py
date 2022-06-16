# %% imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sci


# %%
def f(t, v, m, k, ms, ts, ks):
    # kai kūnai kartu
    if t < ts:
        return (-ms * g - ks * v * abs(v)) / ms
    # kai kūnai atsiskyrę
    else:
        return (-m * g - k * v * abs(v)) / m


# konstantos
g = 9.8
m1 = 0.4
m2 = 0.8
v0 = 50
ks = 0.001
ts = 2
k1 = 0.02
k2 = 0.02
CONST_t_max = 10
CONST_m_sum = 1.2
h0 = 0
dt = 0.025
t0 = 0


# eulerio algoritmas
def Eulers(dt):
    t_array = np.arange(0, CONST_t_max, dt)
    V1 = np.zeros_like(t_array)
    V2 = np.zeros_like(t_array)
    H1 = np.zeros_like(t_array)
    H2 = np.zeros_like(t_array)
    V1[0] = V2[0] = v0
    H1[0] = H2[0] = h0
    for i in range(1, len(t_array)):
        V1[i] = V1[i - 1] + dt * f(t_array[i - 1], V1[i - 1], m1, k1, CONST_m_sum, ts, ks)
        V2[i] = V2[i - 1] + dt * f(t_array[i - 1], V2[i - 1], m2, k2, CONST_m_sum, ts, ks)
        H1[i] = max(H1[i - 1] + dt * V1[i - 1], 0)
        H2[i] = max(H2[i - 1] + dt * V2[i - 1], 0)
    return V1, V2, H1, H2, t_array


def Plot(x, y, label):
    plt.plot(x, y, label=label)


# %% RK45 algoritmas
def RK4(m, k, t, h, v, dt):
    t_rez = []
    v_rez = []
    h_rez = []
    while CONST_t_max >= t:
        v_rez.append(v)
        h_rez.append(h)
        t_rez.append(t)
        dv_dt1 = f(t, v, m, k, CONST_m_sum, ts, ks)
        v1 = v + dt / 2 * dv_dt1
        dv_dt2 = f(t, v1, m, k, CONST_m_sum, ts, ks)
        v2 = v + dt / 2 * dv_dt2
        dv_dt3 = f(t, v2, m, k, CONST_m_sum, ts, ks)
        v3 = v + dt * dv_dt3
        dv_dt4 = f(t, v3, m, k, CONST_m_sum, ts, ks)
        v = v + dt / 6 * (dv_dt1 + 2 * dv_dt2 + 2 * dv_dt3 + dv_dt4)
        h = max(h + dt * v, 0)
        t += dt
    return t_rez, h_rez, v_rez


# %% main
def ExecuteEulers():
    dt = 0.05
    for i in range(0, 5):
        V1, V2, H1, H2, t_array = Eulers(dt)
        Plot(t_array, V1, 'v1, kai d=' + str(dt))
        dt *= 2
    plt.title('Eulerio metodas')
    plt.legend(loc='best')
    plt.xlabel("laikas ")
    plt.ylabel("greitis (m/s)")
    plt.show()

    dt = 0.05
    for i in range(0, 5):
        V1, V2, H1, H2, t_array = Eulers(dt)
        Plot(t_array, H1, 'h1, kai d=' + str(dt))
        dt *= 2
    plt.title('Eulerio metodas')
    plt.legend(loc='best')
    plt.xlabel("laikas (s)")
    plt.ylabel("aukštis")
    plt.show()

    dt = 0.05
    V1, V2, H1, H2, t_array = Eulers(dt)
    sol1 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m1, k1, CONST_m_sum, ts, ks))
    sol2 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m2, k2, CONST_m_sum, ts, ks))
    plt.plot(sol1.t, sol1.y[0], label='Scipy v1')
    plt.plot(sol2.t, sol2.y[0], label='Scipy v2')
    Plot(t_array, V1, 'v1, kai d=' + str(dt))
    Plot(t_array, V2, 'v2, kai d=' + str(dt))
    plt.title('Eulerio metodas')
    plt.legend(loc='best')
    plt.xlabel("laikas (s)")
    plt.ylabel("greitis (m/s)")
    plt.show()
    Plot(t_array, H1, 'h1, kai d=' + str(dt))
    Plot(t_array, H2, 'h2, kai d=' + str(dt))
    plt.title('Eulerio metodas')
    plt.legend(loc='best')
    plt.xlabel("laikas (s)")
    plt.ylabel("aukštis")
    plt.show()
    print("-------------------------------------------")
    print("Eulerio metodas")
    print("Didžiausias pasiektas aukštis pirmo kūno: " + str(max(H1)) + ", kai laikas = " + str(
        t_array[H1.tolist().index(max(H1))]))
    print("Didžiausias pasiektas aukštis antro kūno: " + str(max(H2)) + ", kai laikas = " + str(
        t_array[H2.tolist().index(max(H2))]))


def ExecuteRK45():
    dt = 0.05
    for i in range(0, 5):
        t_rez1, h_rez1, v_rez1 = RK4(m1, k1, t0, h0, v0, dt)
        Plot(t_rez1, v_rez1, 'v1, kai d=' + str(dt))
        dt *= 2
    plt.title('RK45 metodas')
    plt.xlabel("laikas (s)")
    plt.ylabel("greitis (m/s)")
    plt.legend(loc='best')
    plt.show()

    dt = 0.05
    for i in range(0, 5):
        t_rez1, h_rez1, v_rez1 = RK4(m1, k1, t0, h0, v0, dt)
        Plot(t_rez1, h_rez1, 'h1')
        dt *= 2
    plt.title('RK45 metodas')
    plt.xlabel("laikas (s)")
    plt.ylabel("aukštis")
    plt.legend(loc='best')
    plt.show()

    dt = 0.05
    t_rez1, h_rez1, v_rez1 = RK4(m1, k1, t0, h0, v0, dt)
    t_rez2, h_rez2, v_rez2 = RK4(m2, k2, t0, h0, v0, dt)
    sol1 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m1, k1, CONST_m_sum, ts, ks))
    sol2 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m2, k2, CONST_m_sum, ts, ks))
    plt.plot(sol1.t, sol1.y[0], label='Scipy v1')
    plt.plot(sol2.t, sol2.y[0], label='Scipy v2')
    Plot(t_rez1, v_rez1, 'v1, kai d=' + str(dt))
    Plot(t_rez2, v_rez2, 'v2, kai d=' + str(dt))
    plt.title('RK45 metodas')
    plt.xlabel("laikas (s)")
    plt.ylabel("greitis (m/s)")
    plt.legend(loc='best')
    plt.show()
    Plot(t_rez1, h_rez1, 'h1')
    Plot(t_rez2, h_rez2, 'h2')
    plt.title('RK45 metodas')
    plt.xlabel("laikas (s)")
    plt.ylabel("aukstis")
    plt.legend(loc='best')
    plt.show()
    print("-------------------------------------------")
    print("RK45 metodas")
    print("Didžiausias pasiektas aukštis pirmo kūno: " + str(max(h_rez1)) + ", kai laikas = " + str(
        t_rez1[h_rez1.index(max(h_rez1))]))
    print("Didžiausias pasiektas aukštis antro kūno: " + str(max(h_rez2)) + " kai laikas = " + str(
        t_rez2[h_rez2.index(max(h_rez2))]))
    print("-------------------------------------------")
    sol1 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m1, k1, CONST_m_sum, ts, ks))
    sol2 = sci.solve_ivp(f, (0, CONST_t_max), [v0], args=(m2, k2, CONST_m_sum, ts, ks))
    plt.plot(sol1.t, sol1.y[0], label='Pirmasis kūnas')
    plt.plot(sol2.t, sol2.y[0], label='Antrasis kūnas')
    plt.title('Scipy RK45 metodas')
    plt.xlabel("laikas (s)")
    plt.ylabel("greitis (m/s)")
    plt.legend(loc='best')
    plt.show()


ExecuteEulers()
ExecuteRK45()
# %%
