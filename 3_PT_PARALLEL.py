"""
A thermodynamic investigation script for 2D rigid disc collision simulation 
with multi-core processing enabled.
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

Xin Kai Lee 17/3/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import concurrent.futures as ft
import time


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


def run_simulations(random_speed_range):
    """
    Runs the 2D Many-Rigid-Disc Collision Simulation.

    Parameters:
        random_speed_range (float): The speed range where the component 
            velocity of the balls is generated from a uniform distribution of 
            [-random_speed_range, random_speed_range].
    
    Returns:
        (dict of float): Dictionary containing the average temperature and 
            pressure of the system.
    """
    sim_PT = sim.Simulation(
        N_ball=N_ball,
        r_container=r_container,
        r_ball=r_ball,
        m_ball=m_ball,
        random_speed_range=random_speed_range,
    )
    result = sim_PT.run(
        collisions=collisions, pressure=True, temperature=True, progress=False
    )
    return result


# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_container = 50
N_ball = 100
r_ball = 0.1
collisions = 500
random_speed_ranges = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# -----------------------------------------------------------------------------#

pressures = []
temperatures = []

if __name__ == "__main__":
    print("Starting Simulations")

    t_start = time.perf_counter()

    with ft.ProcessPoolExecutor() as executor:
        results = executor.map(run_simulations, random_speed_ranges)

    t_end = time.perf_counter()

    print(f"Time taken = {round(t_end-t_start,2)}s")

    for result in results:
        pressures.append(result["average pressure"])
        temperatures.append(result["average temperature"])

    # Performing a linear fit
    linear_fit = sp.stats.linregress(x=temperatures, y=pressures)
    arr_fit = np.linspace(np.amin(temperatures), np.amax(temperatures), 1000)

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
    plt.plot(temperatures, pressures, "o", mew=0.5, mec="white")

    plt.title("Pressure vs Temperature")
    plt.ylabel("Temperature /K")
    plt.xlabel("Pressure /Pa")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("End of Script")
