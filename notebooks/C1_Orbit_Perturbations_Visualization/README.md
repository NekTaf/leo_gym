# Orbital Perturbations 


In this first example we will plot different environmental perturbations present in LEO. 

Aims:
+ Visualize LEO perturbations
+ Introduction to LEO-GYM structure 

Objectives:
+ Initialize dynamics and satellite objects for ideal and perturbed orbits 
+ Propagate objects and collect states 
+ Plot results



In this example we will be using the already create **satellite\_base** object:

```python
from leo_gym.satellite.satellite_base import Satellite, SatelliteConfig
```

Its states include the ECI position-velocity (rv) states and a control signal representing the continuous thrust applied over the sample time $\texttt{dt}$.


Every major class in **LEO-GYM** has an equivalent **configuration** class. The configuration class in the input to the satellite object when initialized and provides an organized way of defining and checking input parameters to the satellite object. 

The dynamics class is the second important class at the base of LEO-GYM:

```python
from leo_gym.orbit.dynamics.dynamics import Dynamics, DynamicsConfig
```

Every satellite object requires a DynamicsConfig as input in order to initialize the dynamics class internally. Here satellite parameters and modelling coefficients are defined, along side environmental parameters such as present disturbances such as air drag, third body gravitational perturbations amongst others. 

To start exploring, run the notebook file located in the folder. 