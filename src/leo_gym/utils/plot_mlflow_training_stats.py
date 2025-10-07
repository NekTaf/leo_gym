from leo_gym.utils.matplot_style_cfg import *
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter


def plot_rewards(file_path:str,
                 save_path:str = None
                 )->None:
    
    data = np.loadtxt(file_path) 
    steps = data[:, 2]
    rewards = data[:, 1]

    plt.plot(steps, rewards)
    plt.xlabel("Step")
    plt.ylabel("Mean Episodic Rewards")
    # plt.title("Rewards")
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))  # always show in scientific notation
    plt.gca().xaxis.set_major_formatter(formatter)

    if save_path is not None:
        plt.savefig(os.path.join(save_path,"rewards.png"))    
    else:
        plt.savefig("rewards.png") 

    return 
