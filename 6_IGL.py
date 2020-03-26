"""
A thermodynamic investigation script for 2D rigid disc collision simulation.
A PV diagram is drawn to illustrate the Ideal Gas Law.

Independent Variable:
    V : Volume
    N : Number of balls
    T : Temperature

Dependent Variables:
    P : Pressure

Xin Kai Lee 12/3/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def ideal_gas_law(V, N, T):
    """
    Calculates the ideal gas equation of state.

    Parameters:
        V (float): Volume of the container.
        N (int): Number of gas particles.
        T (float): Temperature.
    
    Returns:
        (float): Pressure.
    """
    kb = 1.38064852e-23
    return V ** -1 * N * kb * T


kb = 1.38064852e-23

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_ball = 0.1
collisions = 500
r_containers = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
N_balls = [50, 100]
random_speed_ranges = [500, 1000]
# -----------------------------------------------------------------------------#

volumes = np.array(r_containers) ** 2 * np.pi
speeds = []
pressures = []
temperatures = []

# Create isotherms using same starting velocities
for N_ball in N_balls:
    for random_speed_range in random_speed_ranges:
        speeds.append(sim.generate_random_vel(N_ball, random_speed_range))

# Running simulations
print("Starting Simulations")

i = 0
for N_ball in N_balls:
    for random_speed_range in random_speed_ranges:
        l_pressure_temp = []
        for r_container in r_containers:
            sim_IGL = sim.Simulation(
                N_ball=N_ball,
                r_container=r_container,
                r_ball=r_ball,
                m_ball=m_ball,
                random_speed_range=random_speed_range,
            )
            sim_IGL.set_vel_ball(speeds[i])
            IGL = sim_IGL.run(collisions=collisions, pressure=True, temperature=True)
            l_pressure_temp.append(IGL["average pressure"])
        i += 1
        temperatures.append(IGL["average temperature"])
        pressures.append(l_pressure_temp)


arr_fit = np.linspace(volumes[0], volumes[-1], 1000)

# Graph Plotting
print("Plotting Graph 1 of 1")

j = 0
plt.figure(num="Ideal Gas Law")
sns.set(context="paper", style="darkgrid", palette="muted")

# Plotting Ideal Gas Law using the input physical parameters
for N_ball in N_balls:
    for random_speed_range in random_speed_ranges:
        legend = f"N = {N_ball}, T = %s K" % (float("%.3g" % temperatures[j]))
        plt.plot(arr_fit, ideal_gas_law(arr_fit, N_ball, temperatures[j]), label=legend)
        j += 1

# Data points from simulation
for i, _ in enumerate(pressures):
    plt.plot(volumes, pressures[i], "o", mec="white", mew=0.5)

plt.title("Ideal Gas Law")
plt.xlabel(r"Volume /$m^2$")
plt.ylabel(r"Pressure /Pa")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")

