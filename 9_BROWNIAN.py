"""
A Statistical Physics investigation script to visualise, measure and characterise Brownian Motion in the 2D Many-Rigid-Disc Collision Simulation.

Aspects investigated:
    1. Brownian Motion animation and real-time path tracing.
    2. Recording Brownian Motion Dataset.
    3. Tracing Paths given a Dataset.
    4. Characterising the Distance Increments of Brownian Motion at regular 
        timesteps.

Xin Kai Lee 17/03/2020
"""
import simulation as sim
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy as sp
import time as tm

# -----------------------------------------------------------------------------#
# The presets provide ideal parameters, but they can be varied
m_ball = 5e-26
r_container = 5
N_ball = 50
r_ball = 0.2
r_big = 1
m_big = 2.5e-25
collisions = 500
random_speed_range = 500
# -----------------------------------------------------------------------------#

DATA_PATH = os.path.join(os.getcwd(), "Data")

# First check if a 'Data' folder exists
if os.path.exists(DATA_PATH) == False:
    os.makedirs(DATA_PATH)

# Start Animating Brownian Motion
print("Starting Animation")
print("Plotting Graph 2 of 3")

sim_brownian = sim.Simulation(
    N_ball=N_ball,
    r_container=r_container,
    r_ball=r_ball,
    m_ball=m_ball,
    random_speed_range=random_speed_range,
)
sim_brownian.init_brownian(radius=r_big, mass=m_big)
brownian_test = sim_brownian.run(collisions=collisions, brownian=True, animate=True)[
    "brownian"
]

# Writing Brownian Motion Dataset
print("Writing Dataset generated into .csv file")

fname = f"brownian_{N_ball}_{r_ball}_{m_ball}_{r_container}_{r_big}_{m_big}_{random_speed_range}_{collisions}.csv"
FILE_PATH = os.path.join(DATA_PATH, fname)
brownian_test.to_csv(FILE_PATH)

print(f"File {FILE_PATH} saved")

# Loading pre-run file with 2000000 collisions
print("Loading Brownian Motion Dataset: 2000000 collisions with 400 balls")

FILE_PATH = "Dataset/brownian_400_0.1_5e-26_5_0.5_2.5e-25_500_2000000.csv"
brownian = pd.read_csv(FILE_PATH)

print(f"File loaded: {FILE_PATH}")

# Tracing the path of Brownian Motion
print("Tracing Paths")
l_path = sim.trace_paths_brownian(brownian)

print("Plotting Graph 2 of 3")

plt.figure(num="Brownian Motion")
plt.rcParams.update(plt.rcParamsDefault)

ax = plt.axes(
    xlim=(-r_container, r_container), ylim=(-r_container, r_container), aspect="equal",
)
ax.add_patch(plt.Circle(np.array([0, 0]), r_container, ec="b", fill=False, ls="solid"))

t_start = tm.time()
for i, path in enumerate(l_path):
    sim.progress_bar(t_start, i + 1, len(l_path), desc="Path Drawn")
    ax.add_line(path)

plt.title("Brownian Motion")
plt.tight_layout()
plt.show()

# Plotting the distance increment distribution of Brownian Motion
print("Calculating Distance Increments")

l_increments_x = []
l_increments_y = []

brownian_equal = sim.equal_sampling_brownian(brownian, 8000)
brownian_x = brownian_equal["x"]
brownian_y = brownian_equal["y"]

for i in range(1, len(brownian_x)):
    l_increments_x.append(brownian_x[i] - brownian_x[i - 1])

for i in range(1, len(brownian_y)):
    l_increments_y.append(brownian_y[i] - brownian_y[i - 1])

# Normal Distribution Fit
fit_x = sp.stats.norm.fit(l_increments_x)
fit_y = sp.stats.norm.fit(l_increments_y)

arr_fit = np.linspace(
    np.amin([np.amin(l_increments_x), np.amin(l_increments_y)]),
    np.amax([np.amax(l_increments_x), np.amax(l_increments_y)]),
    1000,
)

pdf_x = sp.stats.norm(*fit_x)
pdf_y = sp.stats.norm(*fit_y)

ks_x = sp.stats.kstest(rvs=l_increments_x, cdf="norm", args=fit_x)
ks_y = sp.stats.kstest(rvs=l_increments_y, cdf="norm", args=fit_y)

print(
    f"Normal Distribution for x increments: KS Statistic = {round(ks_x[0],6)}, p = {round(ks_x[1],3)}"
)

print(
    f"Normal Distribution for y increments: KS Statistic = {round(ks_y[0],6)}, p = {round(ks_y[1],3)}"
)


legend_x = r"$\delta l_x$ = (%s $\pm$ %s) m" % (
    float("%.1g" % fit_x[0]),
    float("%.1g" % fit_x[1]),
)
legend_y = r"$\delta l_y$ = (%s $\pm$ %s) m" % (
    float("%.1g" % fit_y[0]),
    float("%.1g" % fit_y[1]),
)


# Plotting Distance Increment Distribution
print("Plotting Graph 3 of 3")

f, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, num="Distance Increments in Brownian Motion"
)
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
sns.set(context="paper", style="darkgrid", palette="muted")
plt.grid(False)

sns.distplot(l_increments_x, ax=axes[0], kde=False, norm_hist=True)
sns.lineplot(arr_fit, pdf_x.pdf(arr_fit), lw=2, ax=axes[0], label=legend_x)
sns.distplot(l_increments_y, ax=axes[1], kde=False, norm_hist=True)
sns.lineplot(arr_fit, pdf_y.pdf(arr_fit), lw=2, ax=axes[1], label=legend_y)

plt.title("Distance Increment Distribution of Brownian Motion")
plt.xlabel(r"Distance Increments $\delta l$ /$m$")
plt.ylabel(r"Probability Distribution /$m^{-1}$")
plt.tight_layout()
plt.show()

print("End of Script")

