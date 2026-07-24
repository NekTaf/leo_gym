
<h1>
  <img src="images/atellit%20(3).gif" alt="logo" width="100" style="vertical-align: middle; margin-left: 20px;">
  Low Earth Orbit (LEO)-GYM
</h1>


## Motivation 
LEO-GYM is a library for creating Reinforcement Learning (RL) environments for training policies to control satellites operating in LEO.



## Contents

1. **Library Introduction Tutorial**

2. **Across-Track (ACT) Maneuvers in LEO**
   1. Training of a policy to conduct ACT maneuvers in LEO  
   2. Reformulation of ACT maneuvers as a Semi-MDP to account for variable duration

3. **Collision Avoidance Maneuvers**


## Installation

It is recommended to first create a virtual environment. Then clone and install the library (use -e for editable mode):

```bash 
git clone https://github.com/NekTaf/leo_gym.git
cd leo_gym
pip install -e .
nbstripout --install
nbdime config-git --enable
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/NekTaf/leo_gym.git
```
## Acknowledgements
This work has been partially funded by the European Space Agency (ESA) open Invitations to Tender (ITT) and innovation research grant in OPTACOM project, in collaboration with OHB Sweden under Grant Contract no: OPC-OSE-CC-0536

## Citation
If you use LEO-GYM in your research, please cite it as:

```bibtex
@article{tafanidis2025leo,
  title={LEO-GYM: A Reinforcement Learning Library for Satellite Control in LEO},
  author={Tafanidis, Nektarios Aristeidis and Banerjee, Avijit and Nikolakopoulos, George},
  journal={IFAC-PapersOnLine},
  volume={59},
  number={31},
  pages={127--132},
  year={2025},
  publisher={Elsevier}
}
```

## Contact

Nektarios Aristeidis Tafanidis: mail@natafanidis.com


