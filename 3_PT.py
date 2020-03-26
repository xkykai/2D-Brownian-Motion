"""
A thermodynamic investigation script for 2D rigid disc collision simulation.
The speed range of the balls are varied to obtain different temperatures as 
independent variable. The average pressure of the system is obtained. 
The graph of Pressure against Temperature should have a linear relationship given by the ideal gas law where gradient = N * kb / V.

Independent Variable:
    T : Temperature

Dependent Variables:
    P : Pressure

Constants:
    N : Number of balls
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


# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_container = 50
N_ball = 100
r_ball = 0.1
collisions = 500
l_random_speed_range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# -----------------------------------------------------------------------------#

l_pressure = []
l_temperature = []

print("Starting Simulations")

# Running simulations varying the range of speeds
for random_speed_range in l_random_speed_range:
    sim_PT = sim.Simulation(
        N_ball=N_ball,
        r_container=r_container,
        r_ball=r_ball,
        m_ball=m_ball,
        random_speed_range=random_speed_range,
    )
    PT = sim_PT.run(collisions=collisions, pressure=True, temperature=True)
    l_pressure.append(PT["average pressure"])
    l_temperature.append(PT["average temperature"])

# Performing a linear fit
linear_fit = sp.stats.linregress(x=l_temperature, y=l_pressure)
arr_fit = np.linspace(l_temperature[0], l_temperature[-1], 1000)

legend = (
    fr"gradient = %s $\pm$ %s "
    % (float("%.4g" % linear_fit.slope), float("%.1g" % linear_fit.stderr))
    + r"$kg \, s^{-2} \, K^{-1}$"
)

# Graph Plotting
print("Plotting Graph 1 of 1")

plt.figure(num="Varying Temperature")
sns.set(context="paper", style="darkgrid", palette="muted")

plt.plot(
    arr_fit,
    linear(arr_fit, linear_fit.slope, linear_fit.intercept),
    alpha=0.8,
    lw=2,
    label=legend,
)
plt.plot(l_temperature, l_pressure, "o", mew=0.5, mec="white")

plt.title("Pressure vs Temperature")
plt.ylabel("Temperature /K")
plt.xlabel("Pressure /Pa")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")

