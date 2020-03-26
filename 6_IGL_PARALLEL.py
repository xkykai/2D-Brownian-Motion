"""
A thermodynamic investigation script for 2D rigid diss collision simulation with multi-core processing enabled.
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
import concurrent.futures as ft
import time


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


def run_simulations(parameter):
    """
    Runs the 2D Many-Rigid-Disc Collision Simulation.

    Parameters:
        parameter (list of [N_ball, random_speed_range, r_container, speed]):
            N_ball (int): Number of balls in the system.
            r_container (float): Radius of the container.
            random_speed_range (float): The speed range where the component 
                velocity of the balls is generated from a uniform distribution 
                of [-random_speed_range, random_speed_range].
            speed (list of numpy.ndarray of float): List of ball velocities.
    
    Returns:
        (dict of float): Dictionary containing the average temperature and 
            pressure of the system.
    """
    N_ball = parameter[0]
    random_speed_range = parameter[1]
    r_container = parameter[2]
    speed = parameter[3]
    sim_IGL = sim.Simulation(
        N_ball=N_ball,
        r_container=r_container,
        r_ball=r_ball,
        m_ball=m_ball,
        random_speed_range=random_speed_range,
    )
    sim_IGL.set_vel_ball(speed)
    result = sim_IGL.run(
        collisions=collisions, pressure=True, temperature=True, progress=False
    )
    return result


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

volumes = [np.pi * r ** 2 for r in r_containers]
speeds = []
parameters = []
pressures = []
temperatures = []


if __name__ == "__main__":
    print("Generating Speeds")

    # Create isotherms using same starting velocities
    for N_ball in N_balls:
        for random_speed_range in random_speed_ranges:
            speeds.append(sim.generate_random_vel(N_ball, random_speed_range))

    index = 0
    for N_ball in N_balls:
        for random_speed_range in random_speed_ranges:
            for r_container in r_containers:
                parameters.append(
                    [N_ball, random_speed_range, r_container, speeds[index]]
                )
            index += 1

    print("Starting Simulations")

    t_start = time.perf_counter()

    with ft.ProcessPoolExecutor() as executor:
        results = executor.map(run_simulations, parameters)

    t_end = time.perf_counter()

    print(f"Time taken = {round(t_end-t_start,2)}s")

    for i, result in enumerate(results):
        pressures.append(result["average pressure"])
        if i % len(volumes) == 0:
            temperatures.append(result["average temperature"])

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
            plt.plot(
                arr_fit, ideal_gas_law(arr_fit, N_ball, temperatures[j]), label=legend
            )
            j += 1

    # Data points from simulation
    for i, _ in enumerate(temperatures):
        pressures_temp = []
        for j, _ in enumerate(volumes):
            pressures_temp.append(pressures[i * len(volumes) + j])
        plt.plot(volumes, pressures_temp, "o", mec="white", mew=0.5)

    plt.title("Ideal Gas Law")
    plt.xlabel(r"Volume /$m^2$")
    plt.ylabel(r"Pressure /Pa")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("End of Script")

