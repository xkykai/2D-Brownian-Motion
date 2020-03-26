"""
Visualising and verifying the Maxwell-Boltzmann Distribution for the 2D rigid 
disk collision simulation. The balls are left to collide for an extended period 
of time until the system reaches a steady state.
The speedss of all balls in all collisions are plotted to visualise the speed 
distribution. Using the 2D Maxwell-Boltzmann Distribution, a theoretical curve 
is fitted on it using the physical parameters of the system.

Xin Kai Lee 12/3/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def maxwell_boltzmann_dist(v, T, m):
    """
    Calculates the 2D Maxwell-Boltzmann Distribution.

    Parameters:
        v (numpy.ndarray of float): The speeds of gas particles.
        T (float): Temperature.
        m (float): Mass of gas particles.

    Returns:
        (numpy.ndarray of float): The probability density at v.
    """
    kb = 1.38064852e-23
    return (m * v) / (kb * T) * np.exp((-0.5 * m * v ** 2) / (kb * T))


kb = 1.38064852e-23

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_container = 5
N_ball = 300
r_ball = 0.1
collisions = 10000
random_speed_range = 500
# -----------------------------------------------------------------------------#

# Running simulation
print("Starting Simulation")

sim_MBD = sim.Simulation(
    N_ball=N_ball,
    r_container=r_container,
    r_ball=r_ball,
    m_ball=m_ball,
    random_speed_range=random_speed_range,
)
MBD = sim_MBD.run(collisions=collisions, speed=True, temperature=True)

speeds = np.array(MBD["speed"])
temperature = MBD["average temperature"]

# Drawing Maxwell-Boltzmann Distribution using input physical parameters
arr_MBD = np.linspace(np.amin(speeds), np.amax(speeds), 10000)
MBD_curve = maxwell_boltzmann_dist(arr_MBD, temperature, m_ball)

# Graph Plotting
print("Plotting Graph 1 of 1")

plt.figure(num="Maxwell-Boltzmann Distribution")
sns.set(context="paper", style="darkgrid", palette="muted")

sns.distplot(speeds, label="Simulation Data", bins=30, kde=False, norm_hist=True)
plt.plot(arr_MBD, MBD_curve, label="Maxwell-Boltzmann Distribution", lw=2)

plt.title("2D Maxwell-Boltzmann Distribution")
plt.xlabel(r"Speed /$m s^{-1}$ ")
plt.ylabel(r"Probability Density /$m^{-1} s$")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")
