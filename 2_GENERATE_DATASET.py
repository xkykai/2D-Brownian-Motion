"""
A script showing how the raw datasets of the 2D Many-Rigid-Disc Collision 
Simulation can be generated and saved into a .csv for future analysis.
Using the dataset, the exact states of a simulation at any time can be 
recreated.

Xin Kai Lee 17/03/2020
"""
import simulation as sim
import os
import pandas as pd

DATA_PATH = os.path.join(os.getcwd(), "Data")

# First check if a 'Data' folder exists
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# -----------------------------------------------------------------------------#
# The presets can be varied
m_ball = 5e-26
l_N_ball = [10, 20]
l_r_ball = [2, 4]
l_r_container = [100, 200]
l_random_speed_range = [500, 1000]
collisions = 50
# -----------------------------------------------------------------------------#

# Generating datasets
print("Starting Simulations")

for r_ball in l_r_ball:
    for N_ball in l_N_ball:
        for r_container in l_r_container:
            for random_speed_range in l_random_speed_range:
                fname = f"dataset_{N_ball}_{r_ball}_{r_container}_{m_ball}_{random_speed_range}_{collisions}.csv"
                FILE_PATH = os.path.join(DATA_PATH, fname)

                # First check if the dataset already exists
                if os.path.exists(FILE_PATH):
                    print(
                        f"exists: {N_ball} balls, r_ball = {r_ball}, max speed = {random_speed_range}, r_container = {r_container}, {collisions} collisions"
                    )
                    continue
                else:
                    s = sim.Simulation(
                        N_ball=N_ball,
                        r_container=r_container,
                        r_ball=r_ball,
                        m_ball=m_ball,
                        random_speed_range=random_speed_range,
                    )
                    dataset = s.run(collisions=collisions, dataset=True)["dataset"]
                    dataset.to_csv(FILE_PATH)
                    print(f"Generated {FILE_PATH}")

print("End of Script")

