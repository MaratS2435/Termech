import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

T = np.linspace(0, 10, 1000)
t = sp.Symbol('t')
r = 1 + 1.5 * sp.sin(12*t)
phi = 1.25 * t + 0.2 * sp.cos(12 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)
EP = np.zeros_like(T)
PS = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

for i in np.arange(len(T)):
    if WY[i] > 0:
        Eps = (x - (Vy * (1 + (Vy**2))) / (sp.Abs(Wy)))
        Ps = (y + (1 + (Vy**2)) / (sp.Abs(Wy)))
    else:
        Eps = (x + (Vy * (1 + (Vy**2))) / (sp.Abs(Wy)))
        Ps = (y - (1 + (Vy**2)) / (sp.Abs(Wy)))

    EP[i] = sp.Subs(Eps, t, T[i])
    PS[i] = sp.Subs(Ps, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-300, 300], ylim=[-300, 300])
ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0]+WX[0]], [Y[0], Y[0]+WY[0]], 'g')
RLine, = ax1.plot([0, X[0]], [0, Y[0]], 'k')
PLine, = ax1.plot([X[0], EP[0]], [Y[0], PS[0]], 'm')

ArrowX = np.array([-0.2*4, 0, -0.2*4])
ArrowY = np.array([0.1*4, 0, -0.1*4])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')
WRArrowX, WRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[0], WX[0]))
WArrow, = ax1.plot(WRArrowX+X[0]+WX[0], WRArrowY+Y[0]+WY[0], 'g')
PRArrowX, PRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(EP[0], PS[0]))
PArrow, = ax1.plot(PRArrowX+EP[0], PRArrowY+PS[0], 'm')

ArrowXR = np.array([-0.2, 0, -0.2])
ArrowYR = np.array([0.1, 0, -0.1])
RRArrowX, RRArrowY = Rot2D(ArrowXR, ArrowYR, math.atan2(Y[0], X[0]))
RRArrow, = ax1.plot(RRArrowX+X[0], RRArrowY+Y[0], 'k')

def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    WLine.set_data([X[i], X[i]+WX[i]], [Y[i], Y[i]+WY[i]])
    RLine.set_data([0, X[i]], [0, Y[i]])
    PLine.set_data([X[i], EP[i]], [Y[i], PS[i]])

    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+X[i]+VX[i], RArrowY+Y[i]+VY[i])
    WRArrowX, WRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(WRArrowX+X[i]+WX[i], WRArrowY+Y[i]+WY[i])
    RRArrowX, RRArrowY = Rot2D(ArrowXR, ArrowYR, math.atan2(Y[i], X[i]))
    RRArrow.set_data(RRArrowX+X[i], RRArrowY+Y[i])
    PRArrowX, PRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(EP[i], PS[i]))
    PArrow.set_data(PRArrowX+EP[i], PRArrowY+PS[i])
    
    return P, VLine, VArrow, WLine, WArrow, RLine, RRArrow, PLine, PArrow

anim = FuncAnimation(fig, anima, frames=1000, interval=1, repeat=False)
plt.show()
