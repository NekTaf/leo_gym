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

    plt.figure(figsize=(5,2))

    window = 50
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    moving_std = np.array([
        np.std(rewards[i:i+window]) for i in range(len(rewards) - window + 1)
    ])

    steps_ma = steps[window - 1:]

    line, = plt.plot(steps_ma, moving_avg)
    color = line.get_color()

    plt.fill_between(
        steps_ma,
        moving_avg - moving_std,
        moving_avg + moving_std,
        alpha=0.2,
        color=color,
        rasterized=True   # <-- only change that matters for PGF size
    )

    plt.xlabel("Step")
    plt.ylabel("Mean Episodic Rewards")
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig("rewards.png") 
    plt.savefig("rewards.pgf") 

    return 

os.makedirs("temp", exist_ok=True)
os.chdir("temp")

plot_rewards("./temp/sum_rewards")