from deep_rl_for_swarms.common.mpi_running_mean_std import RunningMeanStd
import deep_rl_for_swarms.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import gym
from deep_rl_for_swarms.common.distributions import make_pdtype

class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.layer_norm = False
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, feat_size, gaussian_fixed_var=True):

        num_hid_layers = len(hid_size)
        neighbor_info = ob_space.dim_rec_o
        nr_rec_obs = neighbor_info[0]
        dim_rec_obs = neighbor_info[1]
        rest = ob_space.dim_flat_o - ob_space.dim_local_o
        dim_flat_obs = ob_space.dim_flat_o

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None,) + ob_space.shape)

        flat_obs_input_layer_0 = tf.slice(ob, [0, 0], [-1, nr_rec_obs * dim_rec_obs])
        flat_obs_input_layer_1 = tf.slice(ob, [0, nr_rec_obs * dim_rec_obs], [-1, rest])
        flat_feature_input_layer = tf.slice(ob, [0, nr_rec_obs * dim_rec_obs + rest], [-1, ob_space.dim_local_o])

        with tf.variable_scope('vf'):
            with tf.variable_scope('input_0'):
                input_0_v = tf.layers.dense(flat_obs_input_layer_0, feat_size[0][0], name="fc0",
                                            kernel_initializer=U.normc_initializer(1.0))
            with tf.variable_scope('input_1'):
                input_1_v = tf.layers.dense(flat_obs_input_layer_1, feat_size[1][0], name="fc0",
                                            kernel_initializer=U.normc_initializer(1.0))
            last_out = tf.concat([input_0_v, input_1_v, flat_feature_input_layer], axis=1)
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            with tf.variable_scope('input_0'):
                input_0_pi = tf.layers.dense(flat_obs_input_layer_0, feat_size[0][0], name="fc0",
                                             kernel_initializer=U.normc_initializer(1.0))
            with tf.variable_scope('input_1'):
                input_1_pi = tf.layers.dense(flat_obs_input_layer_1, feat_size[1][0], name="fc0",
                                             kernel_initializer=U.normc_initializer(1.0))
            last_out = tf.concat([input_0_pi, input_1_pi, flat_feature_input_layer], axis=1)
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])
        # self._me = U.function([ob], [me])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

