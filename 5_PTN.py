"""
A thermodynamic investigation script for 2D rigid disk collision simulation.
N, the number of balls in the container is varied and P/T, which is the pressure against temperature of the system is obtained.
The graph of P/T against 1/V should have a linear relationship given by the ideal gas law where gradient = N * kb.

Independent Variable:
    N : Number of balls

Dependent Variables:
    P : Pressure
    T : Temperature

Constants:
    V : Volume

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
r_container = 150
N_balls = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
r_ball = 0.1
collisions = 2000
random_speed_range = 500
# -----------------------------------------------------------------------------#

pressures = []
temperatures = []

# Running simulations varying number of balls
print("Starting Simulations")

for N_ball in N_balls:
    sim_PTN = sim.Simulation(
        N_ball=N_ball,
        r_container=r_container,
        r_ball=r_ball,
        m_ball=m_ball,
        random_speed_range=random_speed_range,
    )
    PTN = sim_PTN.run(collisions=collisions, pressure=True, temperature=True)

    pressures.append(PTN["average pressure"])
    temperatures.append(PTN["average temperature"])

P_over_Ts = np.array(pressures) / np.array(temperatures)

# Linear Fit
linear_fit = sp.stats.linregress(x=N_balls, y=P_over_Ts)
arr_fit = np.linspace(N_balls[0], N_balls[-1], 1000)

legend = (
    fr"gradient = %s $\pm$ %s "
    % (float("%.4g" % linear_fit.slope), float("%.1g" % linear_fit.stderr))
    + r"$kg \, s^{-2} \, K^{-1}$"
)

# Graph Plotting
print("Plotting Graph 1 of 1")

plt.figure(num="Pressure over Temperature against Number Plot")
sns.set(context="paper", style="darkgrid", palette="muted")

plt.plot(
    arr_fit,
    linear(arr_fit, linear_fit.slope, linear_fit.intercept),
    alpha=0.8,
    lw=2,
    label=legend,
)
plt.plot(N_balls, P_over_Ts, "o", mew=0.5, mec="white")

plt.title(r"Pressure vs Number of Particles")
plt.xlabel(r"Number of Particles")
plt.ylabel(r"$\frac{P}{T}$ /$Pa \, K^{-1}$")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")
