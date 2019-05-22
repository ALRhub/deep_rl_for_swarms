import deep_rl_for_swarms.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib as tfc


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, layer_norm=False):

        num_layers = len(hidden_sizes)
        # input_ph holds an unspecified number of flattened inputs to the mean embedding
        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))  # first reshape the input to have only received observations in 2nd dim
        data_input_layer = tf.slice(reshaped_input, [0, 0], [-1, dim_obs - 2])  # grab data part
        valid_input_layer = tf.slice(reshaped_input, [0, dim_obs - 2], [-1, 1])  # grab valid part in case we don't see all agents
        valid_indices = tf.where(tf.cast(valid_input_layer, dtype=tf.bool))[:, 0:1]  # check where the tensor contains data from other agents
        valid_data = tf.gather_nd(data_input_layer, valid_indices)  # take only actual data and put it into new tensor (rest contains only zeros)

        last_out = valid_data

        if num_layers > 0:  # if num_layers == 0, it is a simple mean of the input data without embedding
            for i in range(num_layers):
                last_out = tf.layers.dense(last_out, hidden_sizes[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            fc_out = last_out

            # reverse gather operation: project processed data back into places where they should be
            # output has same number of rows as data_input_layer and hidden_sizes[i] columns
            last_out_scatter = tf.scatter_nd(valid_indices, fc_out,
                                             shape=tf.cast(
                                                 [tf.shape(data_input_layer)[0], tf.shape(fc_out)[1]],
                                                 tf.int64))

            # now we have to reshape the embedded data into 3 dimensions.
            # 1: nr of processed data points (can be number of agents during inference or higher during training)
            # 2: number of received data points per agent (mostly n_agents -1, or more if allocated for added agents)
            # 3: the dimension of the mean embedding
            reshaped_output = tf.reshape(last_out_scatter, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            # same reshape as before, but only with unprocessed data
            reshaped_output = tf.reshape(data_input_layer, shape=(-1, nr_obs, dim_obs - 2))

        # match shape of mean embedding output
        reshaped_nr_obs_var = tf.reshape(valid_input_layer, shape=(-1, nr_obs, 1))

        # simple hack to avoid division by zero in case an agent doesn't observe any other agents
        n = tf.maximum(tf.reduce_sum(reshaped_nr_obs_var, axis=1, name="nr_agents_test"), 1)

        # now we sum over the second dimension...
        last_out_sum = tf.reduce_sum(reshaped_output, axis=1)
        # ... and divide by the number of received data points
        last_out_mean = tf.divide(last_out_sum,
                                  n,
                                  )
        # mean embedding should be [None, hidden_dim[-1] now
        self.me_out = last_out_mean
