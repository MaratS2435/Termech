import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
import sympy as sp


def Rot2D(x, y, Alpha):
    RX = x * np.cos(Alpha) - y * np.sin(Alpha)
    RY = x * np.sin(Alpha) + y * np.cos(Alpha)
    return RX, RY


# time
t = sp.Symbol('t')

# 1000 dots in (1, 15)
T = np.linspace(1, 15, 1000)

Radius = 4

r = 1 + 1.5 * sp.sin(12 * t)
phi = 1.25 * t + 0.2 * sp.cos(12 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
V = sp.sqrt(Vx ** 2 + Vy ** 2)


R = np.zeros_like(T)
PHI = np.zeros_like(T)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)

# заполнение массивов
for i in np.arange(len(T)):
    R[i] = sp.Subs(r, t, T[i])
    PHI[i] = sp.Subs(phi, t, T[i])
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])

# origin
XX = [0 for i in range(1000)]
YY = [0 for i in range(1000)]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-2.5, 2.5], ylim=[-3, 3])
ax1.plot(X, Y)

# point
P, = ax1.plot(X[0], Y[0], 'r', marker='o')

# рисуем вектор в нулевой момент времени (в anima будем рисовать с 1 до последнего момента времени)
v_line, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
v_line3, = ax1.plot([XX[0], X[0]], [YY[0], Y[0]], 'b')


ArrowX = np.array([-0.2 * Radius, 0, -0.2 * Radius])
ArrowY = np.array([0.1 * Radius, 0, -0.1 * Radius])
ArrowWX = np.array([-Radius, 0, -Radius])
ArrowWY = np.array([Radius, 0, -Radius])
ArrowRX = np.array([-0.1 * Radius, 0, -0.1 * Radius])
ArrowRY = np.array([0.05 * Radius, 0, -0.05 * Radius])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RArrowRX, RArrowRY = Rot2D(ArrowRX, ArrowRY, math.atan2(X[0], Y[0]))
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0], 'r')
RArrow, = ax1.plot(ArrowRX + X[0], ArrowRY + Y[0], 'b')


# рисуем в каждый момент времени i нужные нам вектора
def anima(j):
    P.set_data(X[j], Y[j])
    v_line.set_data([X[j], X[j] + VX[j]], [Y[j], Y[j] + VY[j]])
    v_line3.set_data([XX[j], X[j]], [YY[j], Y[j]])
    r_arrow_x, r_arrow_y = Rot2D(ArrowX, ArrowY, math.atan2(VY[j], VX[j]))
    VArrow.set_data(r_arrow_x + X[j] + VX[j], r_arrow_y + Y[j] + VY[j])
    r_arrow_rx, r_arrow_ry = Rot2D(ArrowRX, ArrowRY, math.atan2(Y[j], X[j]))
    RArrow.set_data(r_arrow_rx + X[j], r_arrow_ry + Y[j])

    return P, v_line, VArrow, v_line3, RArrow


anim = FuncAnimation(fig, anima, frames=1000, interval=20, blit=True, repeat=True)
plt.show()