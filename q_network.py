import tensorflow as tf


class QNetwork:

    def __init__(self, state_dim, action_dim, learning_rate=1e-4, tau=0.001, num_hidden_1=400, num_hidden_2=300):

        self.regularization = 0.01
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.weights_initializer = tf.keras.initializers.he_uniform()

        self.state, self.action, self.valid_flag, self.q_value, self.opt_action, self.dqn_variables, self.q_out \
            = self._build_network("dqn")

        self.state_target, self.action_target, _, self.q_value_target, _, self.dqn_variables_target, _ \
            = self._build_network("dqn_target")

        self.target = tf.placeholder(tf.float32, [None])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.dqn_variables if 'bias' not in v.name])

        self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + self.regularization * self.l2_loss
        self.optimize = self.optimizer.minimize(self.loss)

        self.update_target_op = [self.dqn_variables_target[i].assign(
            tf.multiply(self.dqn_variables[i], tau) + tf.multiply(self.dqn_variables_target[i], 1 - tau)) for i in
            range(len(self.dqn_variables))]
        self.init_target_op = [self.dqn_variables_target[i].assign(self.dqn_variables[i])
                               for i in range(len(self.dqn_variables))]

    def _build_network(self, name):
        state = tf.placeholder(tf.float32, [None, self.state_dim], name=name + "_" + "state")
        action = tf.placeholder(tf.int32, [None], name=name + "_" + "action")
        valid_flag = tf.placeholder(tf.float32, [None, self.action_dim])
        action_one_hot = tf.one_hot(action, self.action_dim)

        with tf.variable_scope(name):
            layer_1 = tf.contrib.layers.fully_connected(state, self.num_hidden_1,
                                                        weights_initializer=self.weights_initializer)
            layer_2 = tf.contrib.layers.fully_connected(layer_1, self.num_hidden_2,
                                                        weights_initializer=self.weights_initializer)

            q_out = tf.contrib.layers.fully_connected(layer_2, self.action_dim, activation_fn=None,
                                                      weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

            max_action = tf.argmin(q_out + (1 - valid_flag)*tf.constant(1e8), 1)
            q_value = tf.reduce_sum(tf.multiply(q_out, action_one_hot), axis=1)

        dqn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return state, action, valid_flag, q_value, max_action, dqn_variables, q_out

    def get_q_value_target(self, state, action, sess):
        return sess.run(self.q_value_target, feed_dict={
            self.state_target: state,
            self.action_target: action
        })

    def get_q_value(self, state, action, sess):
        return sess.run(self.q_value, feed_dict={
            self.state: state,
            self.action: action
        })

    def get_opt_action(self, state, valid_flag, sess):
        return sess.run(self.opt_action, feed_dict={
            self.state: state,
            self.valid_flag: valid_flag
        })

    def get_q_out(self, state, sess):
        return sess.run(self.q_out, feed_dict={
            self.state: state
        })

    def train(self, state, action, target, sess):
        _, loss = sess.run([self.optimize, self.loss], feed_dict={
            self.state: state,
            self.action: action,
            self.target: target
        })
        return loss

    def update_target(self, sess):
        sess.run(self.update_target_op)

    def init_target(self, sess):
        sess.run(self.init_target_op)
