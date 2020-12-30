import numpy as np


class AgentDVN:
    def __init__(self, VN, k, gamma=1):
        self.VN = VN
        self.gamma = gamma
        self.k = k

    def get_action_one_step(self, cur_node, path, env, sess, mask_penalty=1e3):
        candidates = env.all_candidates[cur_node]
        poss_next_state = env.all_state[candidates]
        done = env.all_poss_done[cur_node]

        reward = env.link_delay[cur_node, candidates] + env.que_delay[candidates] * (1 - done)

        next_state_value = self.VN.get_value(poss_next_state, sess)

        n = len(done)
        mask_0 = np.ones(n)
        for i in range(n):
            if candidates[i] in path:  # Add a penalty to previously selected node
                mask_0[i] = mask_penalty

        next_state_value = next_state_value * mask_0
        action_value = reward + self.gamma * next_state_value * (1 - done)
        action = np.argmin(action_value)
        return action

    def get_action_train(self, poss_next_state, poss_reward, poss_done, sess):
        next_state_value = self.VN.get_value(poss_next_state, sess)
        q_value = poss_reward + (1 - poss_done) * next_state_value
        action = np.argmin(q_value)

        return action

    def learn_batch(self, exp_rep, state_dim, sess):
        batch = exp_rep.sample()
        if len(batch) == 0:
            return 0

        poss_next_state_batch = np.vstack([v[2] for v in batch])
        split_index = np.cumsum([len(v[1]) for v in batch])
        poss_next_value_batch = self.VN.get_value(np.reshape(poss_next_state_batch, [-1, state_dim]), sess)
        poss_next_value_batch = np.split(poss_next_value_batch, split_index)[:-1]

        state_batch = []
        opt_next_reward_batch = []
        opt_next_state_batch = []
        opt_next_done_batch = []
        for [cur_state, poss_reward, poss_next_state, poss_done], poss_next_value in zip(batch, poss_next_value_batch):
            state_batch.append(cur_state)
            q_value = poss_reward + (1 - poss_done) * poss_next_value[:len(poss_done)]
            opt_action = np.argmin(q_value)
            opt_next_reward_batch.append(poss_reward[opt_action])
            opt_next_state_batch.append(poss_next_state[opt_action])
            opt_next_done_batch.append(poss_done[opt_action])

        next_value_target_batch = self.VN.get_value_target(opt_next_state_batch, sess)

        y_batch = opt_next_reward_batch + next_value_target_batch * (1 - np.array(opt_next_done_batch))

        loss = self.VN.train(state_batch, y_batch, sess)
        self.VN.update_target(sess)
        return loss
