import numpy as np


class Logger:
    def __init__(self, name, results_dir, save_every_num, max_step, save_loss=False):
        self.save_loss = save_loss
        self.save_every_num = save_every_num
        self.name = name
        self.dir = results_dir
        self.max_step = max_step
        self.cum_reward = 0
        self.cum_return = 0
        self.cum_loss = 0
        self.step_cnt = 0
        self.episode_cnt = 0

        self.loss_list = []
        self.return_list = []
        self.LOSS_list = []
        self.RETURN_list = []

    def add(self, reward, loss, done, t, k):
        self.cum_reward = self.cum_reward + reward
        self.cum_loss = self.cum_loss + loss
        self.step_cnt = self.step_cnt + 1

        if done or t == self.max_step - 1:
            self.episode_cnt = self.episode_cnt + 1
            self.cum_return = self.cum_return + self.cum_reward
            self.return_list.append(self.cum_reward)

            average_loss = self.cum_loss / self.step_cnt
            self.loss_list.append(average_loss)

            self.cum_reward = 0
            self.step_cnt = 0
            self.cum_loss = 0

            if self.episode_cnt % self.save_every_num is 0:
                average_return = self.cum_return/self.save_every_num
                self.cum_return = 0

                print("Monte Carlo {}: Episode {}-{}, {}, loss: {}, average_return: {}".
                      format(k, self.episode_cnt - self.save_every_num + 1,
                             self.episode_cnt, self.name, average_loss, average_return))

                np.save(self.dir + '/return_' + self.name, self.return_list)
                if self.save_loss:
                    np.save(self.dir + '/loss_' + self.name, self.loss_list)

    def mc_summary(self):
        self.LOSS_list.append(self.loss_list)
        self.RETURN_list.append(self.return_list)

        np.save(self.dir + '/return_sum_' + self.name, self.RETURN_list)
        if self.save_loss:
            np.save(self.dir + '/loss_sum_' + self.name, self.LOSS_list)

        self.loss_list = []
        self.return_list = []
        self.episode_cnt = 0


class LoggerOpt:
    def __init__(self, results_dir, save_every_num):
        self.dir = results_dir
        self.save_every_num = save_every_num
        self.cum_return = 0
        self.episode_cnt = 0

        self.return_list = []
        self.RETURN_list = []

    def add(self, cum_reward, k, epsilon):
        self.episode_cnt = self.episode_cnt + 1
        self.cum_return = self.cum_return + cum_reward
        self.return_list.append(cum_reward)

        if self.episode_cnt % self.save_every_num is 0:
            average_return = self.cum_return/self.save_every_num
            self.cum_return = 0

            print("epsilon: {}".format(epsilon))
            print("Monte Carlo {}: Episode {}-{}, optimal, average_return: {}".
                  format(k, self.episode_cnt - self.save_every_num + 1,
                         self.episode_cnt, average_return))

            np.save(self.dir + '/return_opt', self.return_list)

    def mc_summary(self):
        self.RETURN_list.append(self.return_list)

        np.save(self.dir + '/return_sum_opt', self.RETURN_list)

        self.return_list = []
        self.episode_cnt = 0


