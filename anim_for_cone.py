from scipy.integrate import odeint
from numpy import cos, sin 
import pylab as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as ani

from tqdm import tqdm

def get_cart_coords(y):
    x = np.zeros((3, y.shape[1]))
    x[0] = y[0]*sin(al) * cos(y[2])
    x[1] = y[0]*sin(al) * sin(y[2])
    x[2] = y[0]*cos(al)
    return x

def derivative(y, t):
    dy = np.zeros(4)
    dy[0] = y[1]
    dy[1] = y[0] * (sin(al)*y[3])**2 - g*cos(al)
    dy[2] = y[3]
    dy[3] = - 2 * y[1]/y[0] * y[3] 
    
#     v2 = y[1]**2 + (y[0]*sin(al)*dy[2])**2
    dy[1] -= k * (y[1])**2
    dy[3] -= k * (y[0]*y[3]*sin(al))**2
    
    return dy
    
L0 = 1
m = 5
g = 9.81
al = np.pi/6
h0 = (L0**2 / (m**2 * cos(al)*sin(al)**2 * g))**(1./3)
dphi0 = L0 / (m * h0**2*sin(al)**2)
k = 0

k = .4
Y0 = [h0, 0, 0, dphi0]
T, Nt = 100, 10000
t = np.linspace(0, T, Nt)
fps = Nt / T
pbar = tqdm(total=Nt)

def update_lines(i):
    line.set_data(coords[0:2, i])
    line.set_3d_properties(coords[2, i])
    pbar.update(1)
    return line,

sol3 = odeint(derivative, Y0, t)
coords = get_cart_coords(sol3.T)


# Attaching 3D axis to the figure
plt.close('anim')
fig = plt.figure('anim')
ax = p3.Axes3D(fig)

line = ax.plot(coords[0, 0], coords[1, 0], coords[2, 0], 'ko')[0]

ax.set_xlim3d([-.15, .15])
ax.set_ylim3d([-.15, .15])
ax.set_zlim3d([0.0, 0.225])

# plot a cone
# Set up the grid in polar
phi = np.linspace(0,2*np.pi,90)
r = np.linspace(0,h0*sin(al),50)
P, R = np.meshgrid(phi, r)

# Then calculate X, Y, and Z
X = R * np.cos(P)
Y = R * np.sin(P)
Z = R/np.tan(al)

ax.plot_wireframe(X, Y, Z, color="k", alpha=0.1, rstride=2, cstride=2)
ax.set_axis_off()
line_ani = ani.FuncAnimation(fig, update_lines, frames=Nt, blit=True)
line_ani.save('test.mp4', fps=fps)