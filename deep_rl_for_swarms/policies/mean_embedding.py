import deep_rl_for_swarms.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib as tfc


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, layer_norm=False):

        num_layers = len(hidden_sizes)
        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))
        data_input_layer = tf.slice(reshaped_input, [0, 0], [-1, dim_obs - 2])
        valid_input_layer = tf.slice(reshaped_input, [0, dim_obs - 2], [-1, 1])
        valid_indices = tf.where(tf.cast(valid_input_layer, dtype=tf.bool))[:, 0:1]
        valid_data = tf.gather_nd(data_input_layer, valid_indices)

        last_out = valid_data

        if num_layers > 0:
            for i in range(num_layers):
                last_out = tf.layers.dense(last_out, hidden_sizes[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            fc_out = last_out

            last_out_scatter = tf.scatter_nd(valid_indices, fc_out,
                                             shape=tf.cast(
                                                 [tf.shape(data_input_layer)[0], tf.shape(fc_out)[1]],
                                                 tf.int64))

            reshaped_output = tf.reshape(last_out_scatter, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            reshaped_output = tf.reshape(data_input_layer, shape=(-1, nr_obs, dim_obs - 2))

        reshaped_nr_obs_var = tf.reshape(valid_input_layer, shape=(-1, nr_obs, 1))

        n = tf.maximum(tf.reduce_sum(reshaped_nr_obs_var, axis=1, name="nr_agents_test"), 1)

        last_out_sum = tf.reduce_sum(reshaped_output, axis=1)
        last_out_mean = tf.divide(last_out_sum,
                                  n,
                                  )

        self.me_out = last_out_mean
