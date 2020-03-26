"""
A thermodynamic investigation script for 2D rigid disc collision simulation.
The volume(area) of the container is varied and P/T, which is the pressure 
against temperature of the system is obtained.
The graph of P/T against 1/V should have a linear relationship given by the ideal gas law where gradient = N * kb.

Independent Variable:
    V : Volume

Dependent Variables:
    P : Pressure
    T : Temperature

Constants:
    N : Number of balls

Xin Kai Lee 12/3/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def linear(x, m, c):
    """
    Calculates a linear function.

    Parameters:
        x (numpy.ndarray of float): x-values.
        m (float): Gradient.
        c (float): y-intercept.
    
    Returns:
        (numpy.ndarray of float): y-values of the linear function.
    """
    return m * x + c


kb = 1.38064852e-23

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_containers = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
N_ball = 100
r_ball = 0.1
collisions = 500
random_speed_range = 500
# -----------------------------------------------------------------------------#


pressures = []
reciprocal_volumes = []
temperatures = []


# Running simulations varying radius of container
print("Starting Simulation")

for r_container in r_containers:
    sim_PTV = sim.Simulation(
        N_ball=N_ball,
        r_container=r_container,
        r_ball=r_ball,
        m_ball=m_ball,
        random_speed_range=random_speed_range,
    )
    PTV = sim_PTV.run(collisions=collisions, pressure=True, temperature=True)

    pressures.append(PTV["average pressure"])
    reciprocal_volumes.append(1 / (r_container ** 2 * np.pi))
    temperatures.append(PTV["average temperature"])

P_over_Ts = np.array(pressures) / np.array(temperatures)

# Linear Fit
linear_fit = sp.stats.linregress(x=reciprocal_volumes, y=P_over_Ts)
arr_fit = np.linspace(reciprocal_volumes[0], reciprocal_volumes[-1], 1000)

legend = (
    fr"gradient = %s $\pm$ %s "
    % (float("%.4g" % linear_fit.slope), float("%.1g" % linear_fit.stderr))
    + r"$kg \, m^2 \, s^{-2}$"
)

# Graph Plotting
print("Plotting Graph 1 of 1")

plt.figure(num=r"Varying Volume")
sns.set(context="paper", style="darkgrid", palette="muted")

plt.plot(
    arr_fit,
    linear(arr_fit, linear_fit.slope, linear_fit.intercept),
    alpha=0.8,
    lw=2,
    label=legend,
)
plt.plot(reciprocal_volumes, P_over_Ts, "o", mew=0.5, mec="white")

plt.title(r"Pressure over Temperature vs Reciprocal Volume")
plt.xlabel(r"Reciprocal Volume $\frac{1}{V}$ /$m^{-2}$")
plt.ylabel(r"$\frac{P}{T}$ /$Pa \, K^{-1}$")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")
