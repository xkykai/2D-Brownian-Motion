"""
A thermodynamic investigation script for 2D rigid disc collision simulation.
The volume(area) of the container is varied and P/T, which is the pressure 
against temperature of the system is obtained.
The graph of P/T against 1/V should have a linear relationship given by the ideal gas law where gradient = N * kb.

Independent Variable:
    r_ball : Ball Radius
    T : Temperature

Dependent Variables:
    P : Pressure
    b : Effective Volume(area) occupied by a ball

Constants:
    N : Number of balls

Xin Kai Lee 17/03/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def van_der_waals(T, N, V, b):
    """
    Calculates the van der Waals' Equation of State.

    Parameters:
        T (float): Temperature.
        V (float): Volume of the container.
        N (int): Number of gas particles.
        b (float): The effective area of a gas particle.

    Returns:
        (float): Pressure.
    """
    kb = 1.38064852e-23
    return (N * kb) / (V - N * b) * T


def calculate_b(V_container, N_ball, m):
    """
    Calculates the effective area of a gas particle.

    Parameters:
        V_container(float): Volume of the container.
        N_ball (int): Number of particles in the system.
        m (float): Gradient of the van der Waals' Equation of State of P 
            against T.
        
    Returns:
        (float): The effective volume of a gas particle.
    """
    kb = 1.38064852e-23
    return volume / N_ball - kb / m


def calculate_err_b(m, err_m):
    """
    Calculates the error on the effective area of a gas particle.

    Parameters:
        m (float): Gradient of the van der Waals' Equation of State of P 
            against T.
        err_m (float): The error associated with m.
        
    Returns:
        (float): The error associated with the effective volume of a gas 
            particle.
    """
    kb = 1.38064852e-23
    return kb / m ** 2 * err_m


def power_law(x, n, A, B):
    """
    Calculates a power law function.

    Parameters:
        x (numpy.ndarray of float): x-values
        n (float): The power of x.
        A (float): Scaling factor on x
        B (float): y- shifting factor.
    
    Returns:
        (numpy.ndarray of float): y-values of the power law.
    """
    return A * x ** n + B


kb = 1.38064852e-23

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_container = 10
N_ball = 100
collisions = 3000
r_balls = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
random_speed_ranges = [500, 1000, 1500, 2000]
# -----------------------------------------------------------------------------#

volume = r_container ** 2 * np.pi

arr_b = np.zeros(len(r_balls))
err_b = np.zeros(len(r_balls))
arr_V = np.zeros(len(r_balls))

# Running Simulations by varying ball radius and temperature
print("Starting Simulations")

for i, r_ball in enumerate(r_balls):
    arr_pressure = np.zeros(len(random_speed_ranges))
    arr_temperature = np.zeros(len(random_speed_ranges))
    for j, random_speed_range in enumerate(random_speed_ranges):
        sim_VDW = sim.Simulation(
            N_ball=N_ball,
            r_container=r_container,
            r_ball=r_ball,
            m_ball=m_ball,
            random_speed_range=random_speed_range,
        )
        VDW = sim_VDW.run(collisions=collisions, pressure=True, temperature=True)

        arr_pressure[j] = VDW["average pressure"]
        arr_temperature[j] = VDW["average temperature"]
    fit_linear = sp.stats.linregress(arr_temperature, arr_pressure)

    m = fit_linear.slope
    err_m = fit_linear.stderr
    arr_b[i] = calculate_b(volume, N_ball, m)
    err_b[i] = calculate_err_b(m, err_m)
    arr_V[i] = np.pi * r_ball ** 2

# Power Law Curve Fit
guess_n = 0.5
guess_A = 1
guess_B = 0.5

p0_power = [guess_n, guess_A, guess_B]

fit_power = sp.optimize.curve_fit(power_law, arr_V, arr_b, p0=p0_power, sigma=err_b)

print(f"Power of b = {fit_power[0][0]} +/- {np.sqrt(fit_power[1][0,0])}")

arr_fit = np.linspace(np.amin(arr_V), np.amax(arr_V), 1000)
data_fit_power = power_law(
    np.linspace(np.amin(arr_V), np.amax(arr_V), 1000), *fit_power[0]
)

legend_fit = r"Power Law, $n = %s \pm %s$" % (
    float("%.2g" % fit_power[0][0]),
    float("%.1g" % np.sqrt(fit_power[1][0, 0])),
)


print("Plotting Graph 1 of 1")

plt.figure(num="Power Law of b and V")
sns.set(context="paper", style="darkgrid", palette="muted")

plt.plot(arr_fit, data_fit_power, label=legend_fit, lw=2, alpha=0.8)
plt.plot(arr_V, arr_b, "o", mew=0.5, mec="white")
plt.errorbar(arr_V, arr_b, yerr=err_b, fmt="none", color="black", capsize=3)

plt.title("Power Law Scaling of Ball Effective Area")
plt.xlabel(r"Area of 1 Ball $V_{ball}$ /$m^2$")
plt.ylabel(r"Effective Area of 1 Ball $b$ /$m^2$")
plt.legend()
plt.tight_layout()
plt.show()

print("End of Script")
