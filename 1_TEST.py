"""
A test script to illustrate the behaviour of the 2D Many-Rigid-Disc Collision simulation.

4 Properties are examined:
    1. Simulation Animation.
    2. Steady State Pressure Distribution.
    3. Kinetic Energy Conservation.
    4. Relative Distance Distribution between Balls.
    5. Distance Distribution of Balls from Origin.

Xin Kai Lee 17/03/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

# The presets provide ideal parameters, but they can be varied
m_ball_animate = 5e-26
N_ball_animate = 100
r_ball_animate = 0.2
r_container_animate = 10
random_speed_range_animate = 500
collisions_animate = 500


print("Starting Animation")

sim_test_animate = sim.Simulation(
    N_ball=N_ball_animate,
    r_container=r_container_animate,
    r_ball=r_ball_animate,
    m_ball=m_ball_animate,
    random_speed_range=random_speed_range_animate,
)
_ = sim_test_animate.run(collisions=collisions_animate, animate=True)


# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball_P = 5e-26
N_ball_P = 10
r_ball_P = 0.2
r_container_P = 1
random_speed_range_P = 500
collisions_P = 50000
# -----------------------------------------------------------------------------#


print("Starting Simulation for Steady State Pressure Distribution")

sim_test_P = sim.Simulation(
    N_ball=N_ball_P,
    r_container=r_container_P,
    r_ball=r_ball_P,
    m_ball=m_ball_P,
    random_speed_range=random_speed_range_P,
)
param_test_P = sim_test_P.run(collisions=collisions_P, test_pressure=True)

pressure_test = param_test_P["pressure"]

# Fitting the Steady State Pressure to a Normal Distribution
fit_norm = sp.stats.norm.fit(pressure_test["pressure"])
ks_norm = sp.stats.kstest(pressure_test["pressure"], "norm", fit_norm)
pdf_norm = sp.stats.norm(*fit_norm)

print(
    f"Normal Distribution: KS Statistic = {round(ks_norm[0],6)}, p = {round(ks_norm[1],3)}"
)


arr_norm = np.linspace(
    min(pressure_test["pressure"]), max(pressure_test["pressure"]), 1000
)

legend_norm = r"$\mu$ = %s Pa, $\sigma$ =  %s Pa" % (
    float("%.2g" % fit_norm[0]),
    float("%.1g" % fit_norm[1]),
)


print("Plotting Graph 1 of 4")

plt.figure(num="Steady State Pressure Distribution")
sns.set(context="paper", style="darkgrid", palette="muted")

sns.distplot(pressure_test["pressure"], kde=False, norm_hist=True, rug=True)
plt.plot(arr_norm, pdf_norm.pdf(arr_norm), label=legend_norm, lw=2)

plt.title("Steady State Pressure Distribution")
plt.xlabel(r"Pressure $/Pa$")
plt.ylabel(r"Probability Density $/Pa ^{-1}$")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
N_ball = 50
r_ball = 0.2
r_container = 10
random_speed_range = 500
collisions = 5000
# -----------------------------------------------------------------------------#

print("Starting Simulation for KE Conservation and Distance Distributions")

sim_test = sim.Simulation(
    N_ball=N_ball,
    r_container=r_container,
    r_ball=r_ball,
    m_ball=m_ball,
    random_speed_range=random_speed_range,
)
param_test = sim_test.run(
    collisions=collisions, KE=True, dist_centre=True, dist_rel=True
)

KE_test = param_test["KE"]
dist_centre_test = param_test["distance from centre"]
dist_rel_test = param_test["relative distance"]


print("Plotting Graph 2 of 4")

plt.figure(num="Kinetic Energy vs Time")
sns.set(context="paper", style="darkgrid", palette="muted")

sns.lineplot("t", "KE", data=KE_test)

plt.title("Kinetic Energy vs Time")
plt.xlabel("Time /s")
plt.ylabel("Kinetic Energy /J")
plt.tight_layout()
plt.show()


print("Plotting Graph 3 of 4")

plt.figure(num="Relative Distance between Balls")
sns.set(context="paper", style="darkgrid", palette="muted")

sns.distplot(dist_rel_test)

plt.title("Relative Ball Distance Distribution")
plt.xlabel(r"Distance between Balls $/m$")
plt.ylabel(r"Probability Density $/m^{-1}$")
plt.tight_layout()
plt.show()


print("Plotting Graph 4 of 4")

plt.figure(num="Distance of Balls from Origin")
sns.set(context="paper", style="darkgrid", palette="muted")

sns.distplot(dist_centre_test, kde=False, norm_hist=True)

plt.title("Distance Distribution from Origin")
plt.xlabel("Ball Distance from Origin $/m$")
plt.ylabel("Probability Density $/m^{-1}$")
plt.tight_layout()
plt.show()

print("End of Script")
