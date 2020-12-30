import os
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)
from environment import AANet
from logger import Logger, LoggerOpt
from agent_dvn import AgentDVN
from agent_dqn import AgentDQN
from value_network import ValueNetwork
from q_network import QNetwork
from gpsr import gpsr_forward, Packet


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: use CPU for training

np.random.seed(1)
save_every_num = 1
plot_path = True
with_que_delay = True

num_flights = 100
max_step = 50
action_dim = 10
state_dim = (action_dim + 2) * 3

delta = 0
lb_lat = 25
ub_lat = 55
lb_lon = -40 + delta
ub_lon = -5 + delta
lb_alt = 9
ub_alt = 12

if plot_path:
    num_test = 100
    initial_R = 1e3
    plot_i = [25]  # list(range(100))
    plot_snapshot = 631
    plot_src_node = 3
    results_dir = 'results/test/plot_path'
else:
    num_test = 1000
    initial_R = 1e8
    plot_i = None
    plot_snapshot = None
    plot_src_node = None
    if with_que_delay:
        results_dir = 'results/test/w_queue'
    else:
        results_dir = 'results/test/wo_queue'

with open('data/test_snapshot.pkl', 'rb') as f:
    test_snapshots = pickle.load(f)

DVN_model_dir = 'results/train/dvn_50/checkpoints/model-1'
DVN_H1 = 50
DVN_H2 = 50

DQN_model_dir = 'results/train/dqn_100/checkpoints/model-1'
DQN_H1 = 100
DQN_H2 = 100

DVN = ValueNetwork(state_dim=state_dim, action_dim=action_dim, num_hidden_1=DVN_H1, num_hidden_2=DVN_H2)
DQN = QNetwork(state_dim=state_dim, action_dim=action_dim, num_hidden_1=DQN_H1, num_hidden_2=DQN_H2)
saver_DVN = tf.train.Saver(tf.trainable_variables('dvn'))
saver_DQN = tf.train.Saver(tf.trainable_variables('dqn'))

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

logger_DVN = Logger('DVN', results_dir, save_every_num, max_step)
logger_DQN = Logger('DQN', results_dir, save_every_num, max_step)
logger_gpsr = Logger('gpsr', results_dir, save_every_num, max_step)
logger_opt = LoggerOpt(results_dir, save_every_num)

agent_DVN = AgentDVN(DVN, action_dim)
agent_DQN = AgentDQN(DQN)

m = Basemap(projection='cyl',
            llcrnrlat=lb_lat,
            urcrnrlat=ub_lat,
            llcrnrlon=lb_lon,
            urcrnrlon=ub_lon,
            resolution='l')
m.drawcoastlines(linewidth=0.25)
m.fillcontinents(zorder=0)
m.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 0], zorder=1, linewidth=0.25)
m.drawmeridians(np.arange(-180, 180, 10), labels=[0, 0, 0, 1], zorder=2, linewidth=0.25)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_DVN.restore(sess, DVN_model_dir)
    saver_DQN.restore(sess, DQN_model_dir)
    i = 0
    tic = time.time()

    while i < num_test:
        if plot_path:
            idx = plot_snapshot
        else:
            idx = np.random.choice(num_test)
        env = AANet(test_snapshots[idx], action_dim, with_que_delay=with_que_delay)
        num_flights = env.num_nodes - 1

        busy_node = env.reset_load()
        poss_initial = env.get_nodes_within_radius(initial_R)
        if plot_path:
            src_node = plot_src_node
        else:
            src_node = np.random.choice(poss_initial)

        while src_node == num_flights:
            src_node = np.random.choice(poss_initial)
        locations = env.all_pos

        # optimal
        _ = env.reset(src_node)
        env.floyd_warshall()
        path_optimal, _ = env.get_opt_path(src_node, num_flights)

        if path_optimal is not None:
            i += 1

            if i in plot_i:
                fig, ax = plt.subplots(figsize=(5, 4))  # figsize=(4, 4)
                m = Basemap(projection='cyl',
                            llcrnrlat=lb_lat,
                            urcrnrlat=ub_lat,
                            llcrnrlon=lb_lon,
                            urcrnrlon=ub_lon,
                            resolution='l')
                m.drawcoastlines(linewidth=0.25)
                m.fillcontinents(zorder=0)

                not_busy_node = [node for node in range(env.num_nodes) if node not in busy_node]
                ax.scatter(locations[not_busy_node, 0], locations[not_busy_node, 1], c='tab:blue', marker='.', zorder=10)
                ax.scatter(locations[busy_node, 0], locations[busy_node, 1], c='black', marker='*', zorder=10)
                last_node = src_node
                for cur_node in path_optimal:
                    l1, = ax.plot([locations[last_node][0], locations[cur_node][0]],
                                  [locations[last_node][1], locations[cur_node][1]],
                                  c='tab:red', linestyle='-', label='Optimal')
                    last_node = cur_node

            logger_opt.add(env.delay[src_node, num_flights], 1, 0)

            # DVN
            cur_node = env.reset(src_node)
            path = [cur_node]
            for t in range(max_step):
                action = agent_DVN.get_action_one_step(cur_node, path, env, sess)

                next_hop = env.all_candidates[cur_node][action]

                last_node = cur_node
                reward, cur_node, done = env.step(next_hop)
                path.append(cur_node)
                if i in plot_i:
                    l2, = ax.plot([locations[last_node][0], locations[cur_node][0]],
                                  [locations[last_node][1], locations[cur_node][1]],
                                  c='tab:green', label='DVN', linestyle='--')

                logger_DVN.add(reward, 0, done, t, 1)
                if done or t == max_step - 1:
                    break

            # DQN
            cur_node = env.reset(src_node)
            path = [cur_node]
            for t in range(max_step):
                state = env.all_state[cur_node]
                action = agent_DQN.get_action_test(cur_node, env, path, sess)

                next_hop = env.all_candidates[cur_node][action]
                last_node = cur_node
                reward, cur_node, done = env.step(next_hop)
                path.append(cur_node)
                if i in plot_i:
                    l3, = ax.plot([locations[last_node][0], locations[cur_node][0]],
                                  [locations[last_node][1], locations[cur_node][1]],
                                  c='tab:cyan', label='DQN', linestyle='-.')
                logger_DQN.add(reward, 0, done, t, 1)
                if done or t == max_step - 1:
                    break

            # GPSR
            cur_node = env.reset(src_node)
            packet = Packet(env.des_node)
            nearest_neighbor = env.nearest_neighbor()
            p_neighbors = env.perimeter_neighbor()
            locations2d = env.all_pos[:, :2]

            for t in range(max_step):
                action = gpsr_forward(packet, cur_node, nearest_neighbor, p_neighbors, locations2d, locations)
                reward, cur_node, done = env.step(action)

                if i in plot_i:
                    l4, = ax.plot([locations[packet.h][0], locations[cur_node][0]],
                                  [locations[packet.h][1], locations[cur_node][1]],
                                  c='tab:orange', linestyle=':', label='GPSR')

                logger_gpsr.add(reward, 0, done, t, 1)
                if done or t == max_step - 1:
                    if i in plot_i:

                        # m.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 0], zorder=1, linewidth=0.25)
                        # m.drawmeridians(np.arange(-180, 180, 10), labels=[0, 0, 0, 1], zorder=2, linewidth=0.25)

                        meridians = np.arange(lb_lon, ub_lon + 5, 5)
                        meridians_label = ['$' + str(x) + '$' for x in meridians]
                        parallels = np.arange(lb_lat, ub_lat + 5, 5)
                        parallels_label = ['$' + str(x) + '$' for x in parallels]

                        ax.set_yticks(parallels)
                        ax.set_yticklabels(parallels_label)
                        ax.set_xticks(meridians)
                        ax.set_xticklabels(meridians_label)
                        ax.set_xlabel(r'Longitude ($^{\circ}$E)')
                        ax.set_ylabel(r'Latitude ($^{\circ}$N)')
                        ax.set_xlim(lb_lon, ub_lon)
                        ax.set_ylim(lb_lat, ub_lat)
                        ax.grid()

                        ax.legend(handles=[l4, l3, l2, l1], loc='center', bbox_to_anchor=(0.5, 0.55), handlelength=2.3)
                        ax.scatter(locations[busy_node, 0], locations[busy_node, 1], c='black', marker='*', zorder=10)
                        fig.tight_layout()
                        fig.savefig(results_dir + '/' + str(i) + '.png')
                        # fig.show()
                    break
            if i % save_every_num is 0:
                print("running time: {}".format(time.time() - tic))
                tic = time.time()
                print("----------------------------------------")
