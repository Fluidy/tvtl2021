import tensorflow as tf


class ValueNetwork:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001, tau=0.001, num_hidden_1=400, num_hidden_2=300):
        self.regularization = 0.01
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.weights_initializer = tf.keras.initializers.he_uniform()

        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2

        self.state, self.value, self.vn_variables = self._build_network('dvn')
        self.state_target, self.value_target, self.vn_variables_target = self._build_network('dvn_target')

        self.target = tf.placeholder(tf.float32, [None])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.vn_variables if 'weights' in v.name])

        self.loss = tf.reduce_mean(tf.square(self.target - self.value)) + self.regularization * self.l2_loss
        self.optimize = self.optimizer.minimize(self.loss)

        self.update_target_op = [self.vn_variables_target[i].assign(
            tf.multiply(self.vn_variables[i], tau) + tf.multiply(self.vn_variables_target[i], 1 - tau)) for i in
            range(len(self.vn_variables))]

        self.init_target_op = [self.vn_variables_target[i].assign(self.vn_variables[i])
                               for i in range(len(self.vn_variables))]

    def _build_network(self, name):
        state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')  # When computing q_value, it should be the next state

        with tf.variable_scope(name):
            layer_1 = tf.contrib.layers.fully_connected(state, self.num_hidden_1,
                                                        weights_initializer=self.weights_initializer)
            layer_2 = tf.contrib.layers.fully_connected(layer_1, self.num_hidden_2,
                                                        weights_initializer=self.weights_initializer)
            value = tf.contrib.layers.fully_connected(layer_2, 1, activation_fn=None,
                                                      weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
            value = tf.squeeze(value, axis=1)

        vn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return state, value, vn_variables

    def get_value(self, state, sess):
        return sess.run(self.value, feed_dict={
            self.state: state
        })

    def get_value_target(self, state, sess):
        return sess.run(self.value_target, feed_dict={
            self.state_target: state
        })

    def update_target(self, sess):
        sess.run(self.update_target_op)

    def train(self, state, target, sess):
        _, loss = sess.run([self.optimize, self.loss], feed_dict={
            self.state: state,
            self.target: target
        })
        return loss
