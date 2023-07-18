# Import all the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba
from numba import jit

# Define necessary parameters
Num_x = 301
Num_t = 100000
dx = 1/(Num_x - 1)
dt = 1e-7
hcot = 1.0
m = 1.0
L = 1
x = np.linspace(0, 1, Num_x)

psi_0 = np.sqrt(2)*np.sin(np.pi*x)
# normal = np.sum(np.absolute(psi_0)**2)*dx

# Potential energy
mu, sigma = 1/2, 1/20
V = -1e4 * np.exp(-(x-mu)**2/(2*sigma**2))

# Draw potential
# plt.figure()
# plt.plot(x, V)
# plt.xlabel("$x$")
# plt.ylabel("$V(x)$")

psi = np.zeros([Num_t, Num_x])
psi[0] = psi_0

@numba.jit("c16[:,:](c16[:,:])", nopython=True, nogil=True)
def compute_psi(psi):
    for t in range(0, Num_t-1):
        for i in range(1, Num_x-1):
            psi[t+1][i] = psi[t][i] + 1j/2 * dt/dx**2 * (psi[t][i+1] - 2*psi[t][i] + psi[t][i-1]) - 1j*dt*V[i]*psi[t][i]
        
        normal = np.sum(np.absolute(psi[t+1])**2)*dx
        for i in range(1, Num_x-1):
            psi[t+1][i] = psi[t+1][i]/normal
        
    return psi
    
psi_calc = compute_psi(psi.astype(complex))

def animate(i):
    line1.set_data(x, np.absolute(psi_calc[100*i])**2)
    # line2.set_data(x, V)
    
fig, ax = plt.subplots(1,1, figsize=(8,4))
line1, = plt.plot([], [], 'r-', lw=2, markersize=8, label="Probability density")
# line2, = plt.plot([], [], 'b-', lw=2, markersize=8, label="Potential")

ax.set_ylim(-1, 20)
ax.set_xlim(0,1)
ax.set_ylabel('$|\psi(x)|^2$', fontsize=20)
ax.set_xlabel('$x/L$', fontsize=20)
plt.tight_layout()

ax.legend(loc='upper left')
ax.set_title('Simulation Probability Density Function $(|\psi(x)|^2)$ of a particle in a box (using Schrodinger Time-Dependent Wave Equation)')

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('probdensity.gif',writer='pillow',fps=50,dpi=100)
plt.show()
