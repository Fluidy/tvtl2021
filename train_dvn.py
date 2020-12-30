import os
import time
import pickle
import numpy as np
import tensorflow as tf
from environment import AANet
from replay import ExpReplay
from logger import Logger, LoggerOpt
from agent_dvn import AgentDVN
from value_network import ValueNetwork
from gpsr import Packet, gpsr_forward

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: use CPU for training

results_dir = 'results/train/dvn_50'

num_monte_carlo = 100

num_hidden_1 = 50
num_hidden_2 = 50
learning_rate = 1e-3

pre_train = 100
anneal = 400
train = 2000
test = 200
num_episodes = pre_train + anneal + train + test

num_flights = 100
max_step = 50
action_dim = 10
state_dim = (action_dim + 2) * 3

save_every_num = 10

start_epsilon = 1
end_epsilon = 0.1

mem_size = 1e5
start_mem = 1000
batch_size = 32
tau = 0.001

DVN = ValueNetwork(state_dim, action_dim, learning_rate, tau, num_hidden_1, num_hidden_2)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(results_dir + '/checkpoints'):
    os.makedirs(results_dir + '/checkpoints')
logger_DVN = Logger('DVN', results_dir, save_every_num, max_step)
logger_opt = LoggerOpt(results_dir, save_every_num)
logger_gpsr = Logger('gpsr', results_dir, save_every_num, max_step)

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)

with open('data/train_snapshot.pkl', 'rb') as f:
    train_snapshots = pickle.load(f)
with open('data/test_snapshot.pkl', 'rb') as f:
    test_snapshots = pickle.load(f)
for k in range(num_monte_carlo):
    exp_rep = ExpReplay(mem_size, start_mem, batch_size)
    agent = AgentDVN(DVN, action_dim)
    epsilon = start_epsilon
    tic = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(DVN.init_target_op)

        i = 0
        while i < num_episodes:
            loss = 0
            idx = np.random.choice(len(train_snapshots))
            env = AANet(train_snapshots[idx], action_dim)

            if pre_train <= i < pre_train + anneal:
                epsilon = epsilon - (start_epsilon - end_epsilon) / anneal

            elif pre_train + anneal <= i < pre_train + anneal + train:
                epsilon = end_epsilon

            elif pre_train + anneal + train <= i:
                epsilon = 0
                idx = np.random.choice(len(test_snapshots))
                env = AANet(test_snapshots[idx], action_dim)

            num_flights = env.num_nodes - 1
            src_node = np.random.choice(num_flights)

            # optimal
            env.reset(src_node)
            env.floyd_warshall()
            path_optimal, _ = env.get_opt_path(src_node, num_flights)  # destination is the last node

            if path_optimal is not None:
                logger_opt.add(env.delay[src_node, num_flights], k + 1, epsilon)
                i += 1

                # DVN
                cur_node = env.reset(src_node)
                for t in range(max_step):
                    cur_state = env.all_state[cur_node]
                    candidates = env.all_candidates[cur_node]
                    poss_reward = env.all_poss_reward[cur_node]
                    poss_next_state = env.all_state[candidates]
                    poss_done = env.all_poss_done[cur_node]
                    exp_rep.add_step([cur_state, poss_reward, poss_next_state, poss_done])

                    if np.random.rand(1) > epsilon:
                        action = agent.get_action_train(poss_next_state, poss_reward, poss_done, sess)
                    else:
                        if env.des_node in candidates:
                            action = np.where(candidates == env.des_node)[0][0]
                        else:
                            action = np.random.choice(len(candidates))

                    next_hop = candidates[action]

                    reward, cur_node, done = env.step(next_hop)

                    if i < pre_train + anneal + train:
                        loss = agent.learn_batch(exp_rep, state_dim, sess)
                        if loss is None:
                            loss = 0
                    logger_DVN.add(reward, loss, done, t, k + 1)
                    if done:
                        break

                # GPSR
                cur_node = env.reset(src_node)
                nearest_neighbor = env.nearest_neighbor()
                p_neighbors = env.perimeter_neighbor()
                packet = Packet(env.des_node)
                locations2d = env.all_pos[:, :2]
                for t in range(max_step):
                    next_hop = gpsr_forward(packet, cur_node, nearest_neighbor, p_neighbors, locations2d, env.all_pos)
                    reward, cur_node, done = env.step(next_hop)
                    logger_gpsr.add(reward, 0, done, t, k + 1)
                    if done:
                        break

            if i % save_every_num is 0:
                print("running time: {}".format(time.time() - tic))
                tic = time.time()
                print("----------------------------------------")

        saver.save(sess, results_dir + '/checkpoints/model', global_step=k+1, write_meta_graph=False)

    logger_DVN.mc_summary()
    logger_opt.mc_summary()
    logger_gpsr.mc_summary()

