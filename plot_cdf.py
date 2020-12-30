import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)
matplotlib.rc('font', family='sans-serif')


def plot_cdf(data_dir, ax, *args, **kwargs):
    x = np.load(data_dir)
    x = np.sort(x)
    y = np.arange(len(x)) / len(x)
    ax.plot(x, y, *args, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])
plot_cdf('results/test/w_queue/return_gpsr.npy', ax, label='GPSR', color='tab:orange', linestyle=':')
plot_cdf('results/test/w_queue/return_DQN.npy', ax, label='DQN', color='tab:cyan', linestyle='-.')
plot_cdf('results/test/w_queue/return_DVN.npy', ax, label='DVN', color='tab:green', linestyle='--')
plot_cdf('results/test/w_queue/return_opt.npy', ax, label='Optimal', color='tab:red')
plot_cdf('results/test/wo_queue/return_gpsr.npy', ax, color='tab:orange', linestyle=':', alpha=0.6)
plot_cdf('results/test/wo_queue/return_DQN.npy', ax, color='tab:cyan', linestyle='-.', alpha=0.6)
plot_cdf('results/test/wo_queue/return_DVN.npy', ax, color='tab:green', linestyle='--', alpha=0.6)
plot_cdf('results/test/wo_queue/return_opt.npy', ax, color='tab:red', alpha=0.6)
ax.legend()
ax.grid()
ax.set_ylabel('CDF')
ax.set_xlabel('E2E Delay (ms)')
ax.set_ylim([0, 1])
ax.set_xlim([0, 500])
ax.legend(handlelength=2.3)
ax.set_xticklabels()
fig.tight_layout()
fig.show()

