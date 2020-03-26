A 2D Circular Rigid Many-Disc Collision Simulation. - Python 3.7.3

Xin Kai Lee 17/03/2020
-------------------------------------------------------------------------------
Important Notes
There are 2 versions to most scripts: X_SCRIPT.py and X_SCRIPT_PARALLEL.py.
X_SCRIPT_PARALLEL.py are multi-core processing enabled. 

I have written a parallelised version for scripts worth parallelising.

It is recommended to run the scripts from the terminal to ensure the custom
progress bar and parallelisation works properly. However, running in iPython 
works just fine too.

When running scripts in terminal, the code will be blocked when the GUI shows 
the graph plotted. Close the GUI window after examining the plot to 
continue running the script.



Module Files
ball.py : Contains the Ball and Container Class and methods for gas particles 
    and container.
event.py : Contains the Event Class for next collision events.
simulation.py : Contains the Simulation Class and methods for simulation.



Class Hierarchy
Simulation has a: Ball, Container, Event
Container is a: Ball
Event is a: tuple



Script Files
1_TEST.py (plots multiple diagrams): Test Script
    Provides the test code to check simulation behaviour.

2_GENERATE_DATASET : Simulation Dataset
    Shows a sample script for generating datasets. Check the 'Data' folder
    after running this script to find the datasets.

3_PT.py, 3_PT_PARALLEL.py : PT Diagram
    Plots a PT Diagram, keeping N and V constant.

4_PTV.py, 4_PT_PARALLEL.py : P/T against V
    Plots a P/T against V diagram, keeping N constant.

5_PTN.py, 5_PTN_PARALLEL.py : P/T against N
    Plots a P/T againt N diagram, keeping V constant.

6_IGL.py, 6_IGL_PARALLEL.py : Ideal Gas Law
    Plots a PV diagram to show the ideal gas law.

7_VDW.py, 7_VDW_PARALLEL.py : Van der Waals' Equation
    Calculates multiple values of b using van der Waals' Equation by fitting 
    many curves on PT diagrams, then plot b against different ball areas.

8_MBD.py : 2D Maxwell-Boltzmann Distribution
    Plots Maxwell-Boltzmann Distribution.

9_BROWNIAN (plots multiple diagrams) : Brownian Motion 
    Generates a dataset for Brownian Motion, and uses a given dataset in 
    "Dataset" folder to trace out the ball path. Characterises Brownian Motion.
