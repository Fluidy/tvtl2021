import numpy as np


class AgentDQN:
    def __init__(self, dqn, gamma=1):
        self.dqn = dqn
        self.gamma = gamma

    def get_action_train(self, state, valid_flag, sess):
        action = self.dqn.get_opt_action([state], [valid_flag], sess)[0]
        return action

    def get_action_test(self, cur_node, env, path, sess, mask_penalty=1e3):
        state = env.all_state[cur_node]

        candidates = env.all_candidates[cur_node]
        q_out = self.dqn.get_q_out([state], sess)[0][:len(candidates)]

        mask = np.ones_like(q_out)
        for i in range(len(candidates)):
            if candidates[i] in path:
                mask[i] = mask_penalty*2

        q_out = q_out * mask

        action = np.argmin(q_out)
        return action

    def learn_batch(self, exp_rep, sess):
        batch = exp_rep.sample()
        if len(batch) == 0:
            return

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_valid_batch = []
        done_batch = []
        for state, action, reward, next_state, next_valid, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_valid_batch.append(next_valid)
            done_batch.append(done)

        next_action_batch = self.dqn.get_opt_action(next_state_batch, next_valid_batch, sess)
        next_q_target_batch = self.dqn.get_q_value_target(next_state_batch, next_action_batch, sess)
        y = np.array(reward_batch) + self.gamma * next_q_target_batch * (1 - np.array(done_batch))
        loss = self.dqn.train(state_batch, action_batch, y, sess)
        self.dqn.update_target(sess)
        return loss



