
import datetime
import numpy as np
import os
import tensorflow as tf
import threading

from ganrl.common.cmd_args import cmd_args
from ganrl.experiment_user_model.data_utils import Dataset
from ganrl.experiment_user_model.utils import UserModelLSTM, UserModelPW


def multithread_compute_vali():
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in xrange(cmd_args.num_thread):
        thread = threading.Thread(target=vali_eval, args=(1, ii))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0]/vali_cnt, vali_sum[1]/vali_cnt, vali_sum[2]/vali_cnt


lock = threading.Lock()


def vali_eval(xx, ii):
    global vali_sum, vali_cnt
    if cmd_args.user_model == 'LSTM':
        vali_thread_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt], feed_dict={user_model.placeholder['clicked_feature']: click_feature_vali[ii],
                                   user_model.placeholder['ut_dispid_feature']: u_t_dispid_feature_vali[ii],
                                   user_model.placeholder['ut_dispid_ut']: np.array(u_t_dispid_split_ut_vali[ii], dtype=np.int64),
                                   user_model.placeholder['ut_dispid']: np.array(u_t_dispid_vali[ii], dtype=np.int64),
                                   user_model.placeholder['ut_clickid']: np.array(u_t_clickid_vali[ii], dtype=np.int64),
                                   user_model.placeholder['ut_clickid_val']: np.ones(len(u_t_clickid_vali[ii]), dtype=np.float32),
                                   user_model.placeholder['click_sublist_index']: np.array(click_sub_index_vali[ii], dtype=np.int64),
                                   user_model.placeholder['ut_dense']: ut_dense_vali[ii],
                                   user_model.placeholder['time']: max_time_vali[ii],
                                   user_model.placeholder['item_size']: news_cnt_short_vali[ii]
                                   })
    elif cmd_args.user_model == 'PW':
        vali_thread_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt],
                                        feed_dict={user_model.placeholder['disp_current_feature']: feature_vali[ii],
                                                user_model.placeholder['item_size']: news_cnt_short_vali[ii],
                                                user_model.placeholder['section_length']: sec_cnt_vali[ii],
                                                user_model.placeholder['click_indices']: np.array(click_2d_vali[ii]),
                                                user_model.placeholder['click_values']: np.ones(len(click_2d_vali[ii]), dtype=np.float32),
                                                user_model.placeholder['disp_indices']: np.array(disp_2d_vali[ii]),
                                                user_model.placeholder['cumsum_tril_indices']: tril_ind_vali[ii],
                                                user_model.placeholder['cumsum_tril_value_indices']: np.array(tril_value_ind_vali[ii], dtype=np.int64),
                                                user_model.placeholder['click_2d_subindex']: click_sub_index_2d_vali[ii],
                                                user_model.placeholder['disp_2d_split_sec_ind']: disp_2d_split_sec_vali[ii],
                                                user_model.placeholder['Xs_clicked']: feature_clicked_vali[ii]})

    lock.acquire()
    vali_sum[0] += vali_thread_eval[0]
    vali_sum[1] += vali_thread_eval[1]
    vali_sum[2] += vali_thread_eval[2]
    vali_cnt += vali_thread_eval[3]
    lock.release()


def multithread_compute_test():
    global test_sum, test_cnt

    num_sets = 1 * cmd_args.num_thread

    thread_dist = [[] for _ in xrange(cmd_args.num_thread)]
    for ii in xrange(num_sets):
        thread_dist[ii % cmd_args.num_thread].append(ii)

    test_sum = [0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in xrange(cmd_args.num_thread):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0]/test_cnt, test_sum[1]/test_cnt, test_sum[2]/test_cnt


def test_eval(xx, thread_dist):
    global test_sum, test_cnt
    test_thread_eval = [0.0, 0.0, 0.0]
    test_thread_cnt = 0
    for ii in thread_dist:
        if cmd_args.user_model == 'LSTM':
            test_set_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt], feed_dict={user_model.placeholder['clicked_feature']: click_feature_test[ii],
                                           user_model.placeholder['ut_dispid_feature']: u_t_dispid_feature_test[ii],
                                           user_model.placeholder['ut_dispid_ut']: np.array(u_t_dispid_split_ut_test[ii], dtype=np.int64),
                                           user_model.placeholder['ut_dispid']: np.array(u_t_dispid_test[ii], dtype=np.int64),
                                           user_model.placeholder['ut_clickid']: np.array(u_t_clickid_test[ii], dtype=np.int64),
                                           user_model.placeholder['ut_clickid_val']: np.ones(len(u_t_clickid_test[ii]), dtype=np.float32),
                                           user_model.placeholder['click_sublist_index']: np.array(click_sub_index_test[ii], dtype=np.int64),
                                           user_model.placeholder['ut_dense']: ut_dense_test[ii],
                                           user_model.placeholder['time']: max_time_test[ii],
                                           user_model.placeholder['item_size']: news_cnt_short_test[ii]
                                           })
        elif cmd_args.user_model == 'PW':
            test_set_eval = sess.run([train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt],
                                        feed_dict={user_model.placeholder['disp_current_feature']: feature_test[ii],
                                                user_model.placeholder['item_size']: news_cnt_short_test[ii],
                                                user_model.placeholder['section_length']: sec_cnt_test[ii],
                                                user_model.placeholder['click_indices']: np.array(click_2d_test[ii]),
                                                user_model.placeholder['click_values']: np.ones(len(click_2d_test[ii]), dtype=np.float32),
                                                user_model.placeholder['disp_indices']: np.array(disp_2d_test[ii]),
                                                user_model.placeholder['cumsum_tril_indices']: tril_ind_test[ii],
                                                user_model.placeholder['cumsum_tril_value_indices']: np.array(tril_value_ind_test[ii], dtype=np.int64),
                                                user_model.placeholder['click_2d_subindex']: click_sub_index_2d_test[ii],
                                                user_model.placeholder['disp_2d_split_sec_ind']: disp_2d_split_sec_test[ii],
                                                user_model.placeholder['Xs_clicked']: feature_clicked_test[ii]})

        test_thread_eval[0] += test_set_eval[0]
        test_thread_eval[1] += test_set_eval[1]
        test_thread_eval[2] += test_set_eval[2]
        test_thread_cnt += test_set_eval[3]

    lock.acquire()
    test_sum[0] += test_thread_eval[0]
    test_sum[1] += test_thread_eval[1]
    test_sum[2] += test_thread_eval[2]
    test_cnt += test_thread_cnt
    lock.release()


if __name__ == '__main__':

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset(cmd_args)

    if cmd_args.resplit:
        dataset.random_split_user()

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start to construct graph" % log_time)

    if cmd_args.user_model == 'LSTM':
        user_model = UserModelLSTM(dataset.f_dim, cmd_args)
    elif cmd_args.user_model == 'PW':
        user_model = UserModelPW(dataset.f_dim, cmd_args)
    else:
        print('using LSTM user model instead.')
        user_model = UserModelLSTM(dataset.f_dim, cmd_args)

    user_model.construct_placeholder()

    train_opt, train_loss, train_prec1, train_prec2, train_loss_sum, train_prec1_sum, train_prec2_sum, train_event_cnt = user_model.construct_model(is_training=True, reuse=False)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, graph completed" % log_time)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # prepare validation data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)

    if cmd_args.user_model == 'LSTM':
        vali_thread_user, size_user_vali, max_time_vali, news_cnt_short_vali, u_t_dispid_vali, \
        u_t_dispid_split_ut_vali, u_t_dispid_feature_vali, click_feature_vali, click_sub_index_vali, \
        u_t_clickid_vali, ut_dense_vali = dataset.prepare_validation_data(cmd_args.num_thread, dataset.vali_user)
    elif cmd_args.user_model == 'PW':
        vali_thread_user, click_2d_vali, disp_2d_vali, \
        feature_vali, sec_cnt_vali, tril_ind_vali, tril_value_ind_vali, disp_2d_split_sec_vali, \
        news_cnt_short_vali, click_sub_index_2d_vali, feature_clicked_vali = dataset.prepare_validation_data(cmd_args.num_thread, dataset.vali_user)
    else:
        vali_thread_user, size_user_vali, max_time_vali, news_cnt_short_vali, u_t_dispid_vali, \
        u_t_dispid_split_ut_vali, u_t_dispid_feature_vali, click_feature_vali, click_sub_index_vali, \
        u_t_clickid_vali, ut_dense_vali = dataset.prepare_validation_data(cmd_args.num_thread, dataset.vali_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    best_metric = [100000.0, 0.0, 0.0]

    vali_path = cmd_args.save_dir+'/'
    if not os.path.exists(vali_path):
        os.makedirs(vali_path)

    saver = tf.train.Saver(max_to_keep=None)

    for i in xrange(cmd_args.num_itrs):
        # training_start_point = (i * cmd_args.batch_size) % (len(dataset.train_user))
        # training_user = dataset.train_user[training_start_point: min(training_start_point + cmd_args.batch_size, len(dataset.train_user))]

        training_user = np.random.choice(dataset.train_user, cmd_args.batch_size, replace=False)

        if cmd_args.user_model == 'LSTM':
            size_user_tr, max_time_tr, news_cnt_short_tr, u_t_dispid_tr, u_t_dispid_split_ut_tr, \
            u_t_dispid_feature_tr, click_feature_tr, click_sub_index_tr, u_t_clickid_tr, ut_dense_tr = dataset.data_process_for_placeholder(training_user)

            sess.run(train_opt, feed_dict={user_model.placeholder['clicked_feature']: click_feature_tr,
                                           user_model.placeholder['ut_dispid_feature']: u_t_dispid_feature_tr,
                                           user_model.placeholder['ut_dispid_ut']: np.array(u_t_dispid_split_ut_tr, dtype=np.int64),
                                           user_model.placeholder['ut_dispid']: np.array(u_t_dispid_tr, dtype=np.int64),
                                           user_model.placeholder['ut_clickid']: np.array(u_t_clickid_tr, dtype=np.int64),
                                           user_model.placeholder['ut_clickid_val']: np.ones(len(u_t_clickid_tr), dtype=np.float32),
                                           user_model.placeholder['click_sublist_index']: np.array(click_sub_index_tr, dtype=np.int64),
                                           user_model.placeholder['ut_dense']: ut_dense_tr,
                                           user_model.placeholder['time']: max_time_tr,
                                           user_model.placeholder['item_size']: news_cnt_short_tr
                                           })
        elif cmd_args.user_model == 'PW':
            click_2d, disp_2d, feature_tr, sec_cnt, tril_ind, tril_value_ind, disp_2d_split_sect, \
            news_cnt_sht, click_2d_subind, feature_clicked_tr = dataset.data_process_for_placeholder(training_user)

            sess.run(train_opt, feed_dict={user_model.placeholder['disp_current_feature']: feature_tr,
                                           user_model.placeholder['item_size']: news_cnt_sht,
                                           user_model.placeholder['section_length']: sec_cnt,
                                           user_model.placeholder['click_indices']: click_2d,
                                           user_model.placeholder['click_values']: np.ones(len(click_2d), dtype=np.float32),
                                           user_model.placeholder['disp_indices']: np.array(disp_2d),
                                           user_model.placeholder['cumsum_tril_indices']: tril_ind,
                                           user_model.placeholder['cumsum_tril_value_indices']: np.array(tril_value_ind, dtype=np.int64),
                                           user_model.placeholder['click_2d_subindex']: click_2d_subind,
                                           user_model.placeholder['disp_2d_split_sec_ind']: disp_2d_split_sect,
                                           user_model.placeholder['Xs_clicked']: feature_clicked_tr})

        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            vali_loss_prc = multithread_compute_vali()
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, vali: %.5f, %.5f, %.5f" %
                  (log_time, i, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

            if vali_loss_prc[0] < best_metric[0]:
                best_metric[0] = vali_loss_prc[0]
                best_save_path = os.path.join(vali_path, 'best-loss')
                best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[1] > best_metric[1]:
                best_metric[1] = vali_loss_prc[1]
                best_save_path = os.path.join(vali_path, 'best-pre1')
                best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[2] > best_metric[2]:
                best_metric[2] = vali_loss_prc[2]
                best_save_path = os.path.join(vali_path, 'best-pre2')
                best_save_path = saver.save(sess, best_save_path)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, iteration %d train complete" % (log_time, i))

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)
    if cmd_args.user_model == 'LSTM':
        test_thread_user, size_user_test, max_time_test, news_cnt_short_test, u_t_dispid_test, \
        u_t_dispid_split_ut_test, u_t_dispid_feature_test, click_feature_test, click_sub_index_test, \
        u_t_clickid_test, ut_dense_test = dataset.prepare_validation_data(cmd_args.num_thread, dataset.test_user)
    elif cmd_args.user_model == 'PW':
        test_thread_user, click_2d_test, disp_2d_test, \
        feature_test, sec_cnt_test, tril_ind_test, tril_value_ind_test, disp_2d_split_sec_test, \
        news_cnt_short_test, click_sub_index_2d_test, feature_clicked_test = dataset.prepare_validation_data(cmd_args.num_thread, dataset.test_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    best_save_path = os.path.join(vali_path, 'best-loss')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test()
    vali_loss_prc = multithread_compute_vali()
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

    best_save_path = os.path.join(vali_path, 'best-pre1')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test()
    vali_loss_prc = multithread_compute_vali()
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

    best_save_path = os.path.join(vali_path, 'best-pre2')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test()
    vali_loss_prc = multithread_compute_vali()
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))
