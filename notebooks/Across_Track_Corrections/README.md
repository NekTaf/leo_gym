# Across-Track Orbital Corrections Policy Training

In this example we will be training a policy to compute finite burn maneuvers and correct the relative orbit of a satellite with respect to an ideal trajectory. 


The finite burn maneuver consists of a:
+ Discrete firing direction 
+ Firing delay: $\Delta t_\text{del}$
+ Firing duration: $\Delta t_\text{dur}$


Two formulations are presented, one "naive" considering reward discounting based on discrete episodic steps and one more correctly formulated as a Semi Markov Decision Process, taking the variable sojourn time $\tau = \Delta t_\text{del} + \Delta t_\text{dur}$


The examples use the RoeGym environment:

```python
from leo_gym.gyms.roe_sk_gym import RoeGym
```

The examples are trained using StableBaselines3 PPO as well as a modified version to account for the variable time action duration. 

The modified SB3 PPO is located at:
```python
from leo_gym.rl_algorithms.ppo_sb3_smdp.ppo_sb3_smdp import SMDP_PPO

```



