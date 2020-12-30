import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)


def plot_learning(result_dir, ax, smooth=50, interval=50, error_bar=True, **kwargs):
    data = np.load(result_dir)
    data = pd.DataFrame(data).rolling(window=smooth, axis=1).mean().values
    mean = data.mean(axis=0)[::interval]
    episode = np.arange(data.shape[1])[::interval]
    ax.semilogy(episode, mean, **kwargs)
    if error_bar:
        std = data.std(axis=0)[::interval]
        if 'color' in kwargs.keys():
            ax.fill_between(episode, mean - std, mean + std,
                            alpha=0.3, facecolor=kwargs['color'])
        else:
            ax.fill_between(episode, mean - std, mean + std,
                            alpha=0.3)


fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('results/train/dvn_50/return_sum_gpsr.npy',
              ax, label='GPSR', linestyle=':', color='tab:orange')
plot_learning('results/train/dqn_100/return_sum_DQN.npy',
              ax, label='DQN', linestyle='-.', color='tab:cyan')
plot_learning('results/train/dvn_50/return_sum_DVN.npy',
              ax, label='DVN', linestyle='--', color='tab:green')
plot_learning('results/train/dvn_50/return_sum_opt.npy',
              ax, label='Optimal', linestyle='-', color='tab:red')


ax.set_xlim([100, 2500])
ax.set_ylim([60, 450])
ax.set_xlabel('Episode')
ax.set_xticks(np.arange(500, 3000, 500))
ax.set_yticks([60, 70, 80, 90, 100, 200, 300, 400])

ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel('E2E Delay (ms)')
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
fig.show()
