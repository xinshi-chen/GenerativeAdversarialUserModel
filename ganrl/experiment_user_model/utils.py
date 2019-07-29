import tensorflow as tf


def mlp(x, hidden_dims, output_dim, activation, sd, act_last=False):
    hidden_dims = tuple(map(int, hidden_dims.split("-")))
    for h in hidden_dims:
        x = tf.layers.dense(x, h, activation=activation, trainable=True,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=sd))
    if act_last:
        return tf.layers.dense(x, output_dim, activation=activation, trainable=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=sd))
    else:
        return tf.layers.dense(x, output_dim, trainable=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=sd))


class UserModelLSTM(object):

    def __init__(self, f_dim, args, max_disp_size=None):

        self.f_dim = f_dim
        self.placeholder = {}
        self.rnn_hidden = args.rnn_hidden_dim
        self.hidden_dims = args.dims
        self.lr = args.learning_rate
        self.max_disp_size = max_disp_size

    def construct_placeholder(self):

        self.placeholder['clicked_feature'] = tf.placeholder(tf.float32, (None, None, self.f_dim))  # (time, user=batch, f_dim)
        self.placeholder['ut_dispid_feature'] = tf.placeholder(tf.float32, shape=[None, self.f_dim])  # # (user*time*dispid, _f_dim)
        self.placeholder['ut_dispid_ut'] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        self.placeholder['ut_dispid'] = tf.placeholder(dtype=tf.int64, shape=[None, 3])
        self.placeholder['ut_clickid'] = tf.placeholder(dtype=tf.int64, shape=[None, 3])
        self.placeholder['ut_clickid_val'] = tf.placeholder(dtype=tf.float32, shape=[None])
        self.placeholder['click_sublist_index'] = tf.placeholder(dtype=tf.int64, shape=[None])

        self.placeholder['ut_dense'] = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.placeholder['time'] = tf.placeholder(dtype=tf.int64)
        self.placeholder['item_size'] = tf.placeholder(dtype=tf.int64)

    def construct_computation_graph(self):

        batch_size = tf.shape(self.placeholder['clicked_feature'])[1]
        denseshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(self.placeholder['time'], [-1]), tf.reshape(self.placeholder['item_size'], [-1])], 0)

        # construct lstm
        cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_hidden, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, self.placeholder['clicked_feature'], initial_state=initial_state, time_major=True)
        # rnn_outputs: (time, user=batch, rnn_hidden)
        # (1) output forward one-step (2) then transpose
        u_bar_feature = tf.concat([tf.zeros([1, batch_size, self.rnn_hidden], dtype=tf.float32), rnn_outputs], 0)
        u_bar_feature = tf.transpose(u_bar_feature, perm=[1, 0, 2])  # (user, time, rnn_hidden)
        # gather corresponding feature
        u_bar_feature_gather = tf.gather_nd(u_bar_feature, self.placeholder['ut_dispid_ut'])
        combine_feature = tf.concat([u_bar_feature_gather, self.placeholder['ut_dispid_feature']], axis=1)
        # indicate size
        combine_feature = tf.reshape(combine_feature, [-1, self.rnn_hidden + self.f_dim])

        # utility
        u_net = mlp(combine_feature, self.hidden_dims, 1, activation=tf.nn.elu, sd=1e-1, act_last=False)
        u_net = tf.reshape(u_net, [-1])

        click_u_tensor = tf.SparseTensor(self.placeholder['ut_clickid'], tf.gather(u_net, self.placeholder['click_sublist_index']), dense_shape=denseshape)
        disp_exp_u_tensor = tf.SparseTensor(self.placeholder['ut_dispid'], tf.exp(u_net), dense_shape=denseshape)  # (user, time, id)
        disp_sum_exp_u_tensor = tf.sparse_reduce_sum(disp_exp_u_tensor, axis=2)
        sum_click_u_tensor = tf.sparse_reduce_sum(click_u_tensor, axis=2)

        loss_tmp = - sum_click_u_tensor + tf.log(disp_sum_exp_u_tensor + 1)  # (user, time) loss
        loss_sum = tf.reduce_sum(tf.multiply(self.placeholder['ut_dense'], loss_tmp))
        event_cnt = tf.reduce_sum(self.placeholder['ut_dense'])
        loss = loss_sum / event_cnt

        dense_exp_disp_util = tf.sparse_tensor_to_dense(disp_exp_u_tensor, default_value=0.0, validate_indices=False)

        click_tensor = tf.sparse_to_dense(self.placeholder['ut_clickid'], denseshape, self.placeholder['ut_clickid_val'], default_value=0.0, validate_indices=False)
        argmax_click = tf.argmax(click_tensor, axis=2)
        argmax_disp = tf.argmax(dense_exp_disp_util, axis=2)

        top_2_disp = tf.nn.top_k(dense_exp_disp_util, k=2, sorted=False)[1]
        argmax_compare = tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32)
        precision_1_sum = tf.reduce_sum(tf.multiply(self.placeholder['ut_dense'], argmax_compare))
        tmpshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(self.placeholder['time'], [-1]), tf.constant([1], dtype=tf.int64)], 0)
        top2_compare = tf.reduce_sum(tf.cast(tf.equal(tf.reshape(argmax_click, tmpshape), tf.cast(top_2_disp, tf.int64)), tf.float32), axis=2)
        precision_2_sum = tf.reduce_sum(tf.multiply(self.placeholder['ut_dense'], top2_compare))
        precision_1 = precision_1_sum / event_cnt
        precision_2 = precision_2_sum / event_cnt

        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt

    def construct_computation_graph_u(self):

        batch_size = tf.shape(self.placeholder['clicked_feature'])[1]

        # construct lstm
        cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_hidden, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, self.placeholder['clicked_feature'], initial_state=initial_state, time_major=True)
        # rnn_outputs: (time, user=batch, rnn_hidden)
        # (1) output forward one-step (2) then transpose
        u_bar_feature = tf.concat([tf.zeros([1, batch_size, self.rnn_hidden], dtype=tf.float32), rnn_outputs], 0)
        u_bar_feature = tf.transpose(u_bar_feature, perm=[1, 0, 2])  # (user, time, rnn_hidden)
        # gather corresponding feature
        u_bar_feature_gather = tf.gather_nd(u_bar_feature, self.placeholder['ut_dispid_ut'])
        combine_feature = tf.concat([u_bar_feature_gather, self.placeholder['ut_dispid_feature']], axis=1)
        # indicate size
        combine_feature = tf.reshape(combine_feature, [-1, self.rnn_hidden + self.f_dim])

        # utility
        u_net = mlp(combine_feature, self.hidden_dims, 1, activation=tf.nn.elu, sd=1e-1, act_last=False)
        self.u_net = tf.reshape(u_net, [-1])
        self.min_trainable_variables = tf.trainable_variables()

    def construct_computation_graph_policy(self):
        batch_size = tf.shape(self.placeholder['clicked_feature'])[1]
        denseshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(self.placeholder['time'], [-1]), tf.reshape(self.placeholder['item_size'], [-1])], 0)

        with tf.variable_scope('lstm2'):
            cell2 = tf.contrib.rnn.BasicLSTMCell(self.rnn_hidden, state_is_tuple=True)
            initial_state2 = cell2.zero_state(batch_size, tf.float32)
            rnn_outputs2, rnn_states2 = tf.nn.dynamic_rnn(cell2, self.placeholder['clicked_feature'], initial_state=initial_state2, time_major=True)

        u_bar_feature2 = tf.concat([tf.zeros([1, batch_size, self.rnn_hidden], dtype=tf.float32), rnn_outputs2], 0)
        u_bar_feature2 = tf.transpose(u_bar_feature2, perm=[1, 0, 2])  # (user, time, rnn_hidden)

        u_bar_feature_gather2 = tf.gather_nd(u_bar_feature2, self.placeholder['ut_dispid_ut'])
        combine_feature2 = tf.concat([u_bar_feature_gather2, self.placeholder['ut_dispid_feature']], axis=1)

        combine_feature2 = tf.reshape(combine_feature2, [-1, self.rnn_hidden + self.f_dim])

        pi_net = mlp(combine_feature2, '256-32', 1, tf.nn.elu, 1e-2)
        pi_net = tf.reshape(pi_net, [-1])

        disp_pi_tensor = tf.SparseTensor(self.placeholder['ut_dispid'], pi_net, dense_shape=denseshape)

        disp_pi_dense_tensor = tf.sparse_add((-10000.0) * tf.ones(tf.cast(denseshape, tf.int32)), disp_pi_tensor)

        disp_pi_dense_tensor = tf.reshape(disp_pi_dense_tensor, [tf.cast(batch_size, tf.int32), tf.cast(self.placeholder['time'], tf.int32), self.max_disp_size])

        pi_net = tf.contrib.layers.softmax(disp_pi_dense_tensor)

        pi_net_val = tf.gather_nd(pi_net, self.placeholder['ut_dispid'])

        loss_max_sum = tf.reduce_sum(tf.multiply(pi_net_val, self.u_net - 0.5 * pi_net_val))
        event_cnt = tf.reduce_sum(self.placeholder['ut_dense'])

        loss_max = loss_max_sum / event_cnt

        sum_click_u_tensor = tf.reduce_sum(tf.gather(self.u_net, self.placeholder['click_sublist_index']))
        loss_min_sum = loss_max_sum - sum_click_u_tensor
        loss_min = loss_min_sum / event_cnt

        click_tensor = tf.sparse_to_dense(self.placeholder['ut_clickid'], denseshape, self.placeholder['ut_clickid_val'], default_value=0.0)
        argmax_click = tf.argmax(click_tensor, axis=2)
        argmax_disp = tf.argmax(pi_net, axis=2)

        top_2_disp = tf.nn.top_k(pi_net, k=2, sorted=False)[1]
        argmax_compare = tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32)
        precision_1_sum = tf.reduce_sum(tf.multiply(self.placeholder['ut_dense'], argmax_compare))
        tmpshape = tf.concat([tf.cast(tf.reshape(batch_size, [-1]), tf.int64), tf.reshape(self.placeholder['time'], [-1]), tf.constant([1], dtype=tf.int64)], 0)
        top2_compare = tf.reduce_sum(tf.cast(tf.equal(tf.reshape(argmax_click, tmpshape), tf.cast(top_2_disp, tf.int64)), tf.float32), axis=2)
        precision_2_sum = tf.reduce_sum(tf.multiply(self.placeholder['ut_dense'], top2_compare))
        precision_1 = precision_1_sum / event_cnt
        precision_2 = precision_2_sum / event_cnt

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        max_trainable_variables = list(set(tf.trainable_variables()) - set(self.min_trainable_variables))

        # lossL2_min = tf.add_n([tf.nn.l2_loss(v) for v in min_trainable_variables if 'bias' not in v.name]) * _regularity
        # lossL2_max = tf.add_n([tf.nn.l2_loss(v) for v in max_trainable_variables if 'bias' not in v.name]) * _regularity
        train_min_op = opt.minimize(loss_min, var_list=self.min_trainable_variables)
        train_max_op = opt.minimize(-loss_max, var_list=max_trainable_variables)

        self.init_variables = list(set(tf.global_variables()) - set(self.min_trainable_variables))

        return train_min_op, train_max_op, loss_min, loss_max, precision_1, precision_2, loss_min_sum, loss_max_sum, precision_1_sum, precision_2_sum, event_cnt

    def construct_model(self, is_training, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt = self.construct_computation_graph()

        if is_training:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step, 100000, 0.96, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = opt.minimize(loss, global_step=global_step)

            return train_op, loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt
        else:
            return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt


class UserModelPW(object):

    def __init__(self, f_dim, args):
        self.f_dim = f_dim
        self.placeholder = {}
        self.hidden_dims = args.dims
        self.lr = args.learning_rate
        self.pw_dim = args.pw_dim
        self.band_size = args.pw_band_size

    def construct_placeholder(self):
        self.placeholder['disp_current_feature'] = tf.placeholder(dtype=tf.float32, shape=[None, self.f_dim])
        self.placeholder['Xs_clicked'] = tf.placeholder(dtype=tf.float32, shape=[None, self.f_dim])

        self.placeholder['item_size'] = tf.placeholder(dtype=tf.int64, shape=[])
        self.placeholder['section_length'] = tf.placeholder(dtype=tf.int64)
        self.placeholder['click_indices'] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        self.placeholder['click_values'] = tf.placeholder(dtype=tf.float32, shape=[None])
        self.placeholder['disp_indices'] = tf.placeholder(dtype=tf.int64, shape=[None, 2])

        self.placeholder['disp_2d_split_sec_ind'] = tf.placeholder(dtype=tf.int64, shape=[None])

        self.placeholder['cumsum_tril_indices'] = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        self.placeholder['cumsum_tril_value_indices'] = tf.placeholder(dtype=tf.int64, shape=[None])

        self.placeholder['click_2d_subindex'] = tf.placeholder(dtype=tf.int64, shape=[None])

    def construct_computation_graph(self):

        denseshape = [self.placeholder['section_length'], self.placeholder['item_size']]

        # (1) history feature --- net ---> clicked_feature
        # (1) construct cumulative history
        click_history = [[] for _ in xrange(self.pw_dim)]
        for ii in xrange(self.pw_dim):
            position_weight = tf.get_variable('p_w'+str(ii), [self.band_size], initializer=tf.constant_initializer(0.0001))
            cumsum_tril_value = tf.gather(position_weight, self.placeholder['cumsum_tril_value_indices'])
            cumsum_tril_matrix = tf.SparseTensor(self.placeholder['cumsum_tril_indices'], cumsum_tril_value,
                                                 [self.placeholder['section_length'], self.placeholder['section_length']])  # sec by sec
            click_history[ii] = tf.sparse_tensor_dense_matmul(cumsum_tril_matrix, self.placeholder['Xs_clicked'])  # Xs_clicked: section by _f_dim
        concat_history = tf.concat(click_history, axis=1)
        disp_history_feature = tf.gather(concat_history, self.placeholder['disp_2d_split_sec_ind'])

        # (4) combine features
        concat_disp_features = tf.reshape(tf.concat([disp_history_feature, self.placeholder['disp_current_feature']], axis=1),
                                          [-1, self.f_dim * self.pw_dim + self.f_dim])

        # (5) compute utility
        u_disp = mlp(concat_disp_features, self.hidden_dims, 1, tf.nn.elu, 1e-3, act_last=False)

        # (5)
        exp_u_disp = tf.exp(u_disp)
        sum_exp_disp_ubar_ut = tf.segment_sum(exp_u_disp, self.placeholder['disp_2d_split_sec_ind'])
        sum_click_u_bar_ut = tf.gather(u_disp, self.placeholder['click_2d_subindex'])

        # (6) loss and precision
        click_tensor = tf.SparseTensor(self.placeholder['click_indices'], self.placeholder['click_values'], denseshape)
        click_cnt = tf.sparse_reduce_sum(click_tensor, axis=1)
        loss_sum = tf.reduce_sum(- sum_click_u_bar_ut + tf.log(sum_exp_disp_ubar_ut + 1))
        event_cnt = tf.reduce_sum(click_cnt)
        loss = loss_sum / event_cnt

        exp_disp_ubar_ut = tf.SparseTensor(self.placeholder['disp_indices'], tf.reshape(exp_u_disp, [-1]), denseshape)
        dense_exp_disp_util = tf.sparse_tensor_to_dense(exp_disp_ubar_ut, default_value=0.0, validate_indices=False)
        argmax_click = tf.argmax(tf.sparse_tensor_to_dense(click_tensor, default_value=0.0), axis=1)
        argmax_disp = tf.argmax(dense_exp_disp_util, axis=1)

        top_2_disp = tf.nn.top_k(dense_exp_disp_util, k=2, sorted=False)[1]

        precision_1_sum = tf.reduce_sum(tf.cast(tf.equal(argmax_click, argmax_disp), tf.float32))
        precision_1 = precision_1_sum / event_cnt
        precision_2_sum = tf.reduce_sum(tf.cast(tf.equal(tf.reshape(argmax_click, [-1, 1]), tf.cast(top_2_disp, tf.int64)), tf.float32))
        precision_2 = precision_2_sum / event_cnt

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.05  # regularity
        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt

    def construct_model(self, is_training, reuse=False):
        global lossL2
        with tf.variable_scope('model', reuse=reuse):
            loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt = self.construct_computation_graph()

        if is_training:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step, 100000, 0.96, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = opt.minimize(loss, global_step=global_step)
            return train_op, loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt
        else:
            return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt
