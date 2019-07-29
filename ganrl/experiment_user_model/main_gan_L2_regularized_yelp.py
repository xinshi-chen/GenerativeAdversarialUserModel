
import datetime
import numpy as np
import tensorflow as tf
import threading
import os

from ganrl.common.cmd_args import cmd_args
from ganrl.experiment_user_model.data_utils import Dataset
from ganrl.experiment_user_model.utils import UserModelLSTM, UserModelPW


def multithread_compute_vali():
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in xrange(cmd_args.num_thread):
        thread = threading.Thread(target=vali_eval, args=(1, ii))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0]/vali_cnt, vali_sum[1]/vali_cnt, vali_sum[2]/vali_cnt, vali_sum[3]/vali_cnt


def vali_eval(xx, ii):
    global vali_sum, vali_cnt
    vali_thread_eval = sess.run([train_loss_min_sum, train_loss_max_sum, train_prec1_sum, train_prec2_sum, train_event_cnt],
                                feed_dict={user_model.placeholder['clicked_feature']: click_feature_vali[ii],
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
    lock.acquire()
    vali_sum[0] += vali_thread_eval[0]
    vali_sum[1] += vali_thread_eval[1]
    vali_sum[2] += vali_thread_eval[2]
    vali_sum[3] += vali_thread_eval[3]
    vali_cnt += vali_thread_eval[4]
    lock.release()


def multithread_compute_test():
    global test_sum, test_cnt

    num_sets = cmd_args.num_thread

    thread_dist = [[] for _ in xrange(cmd_args.num_thread)]
    for ii in xrange(num_sets):
        thread_dist[ii % cmd_args.num_thread].append(ii)

    test_sum = [0.0, 0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in xrange(cmd_args.num_thread):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0]/test_cnt, test_sum[1]/test_cnt, test_sum[2]/test_cnt, test_sum[3]/test_cnt


def test_eval(xx, thread_dist):
    global test_sum, test_cnt
    test_thread_eval = [0.0, 0.0, 0.0, 0.0]
    test_thread_cnt = 0
    for ii in thread_dist:
        test_set_eval = sess.run([train_loss_min_sum, train_loss_max_sum, train_prec1_sum, train_prec2_sum, train_event_cnt],
                                 feed_dict={user_model.placeholder['clicked_feature']: click_feature_test[ii],
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
        test_thread_eval[0] += test_set_eval[0]
        test_thread_eval[1] += test_set_eval[1]
        test_thread_eval[2] += test_set_eval[2]
        test_thread_eval[3] += test_set_eval[3]
        test_thread_cnt += test_set_eval[4]

    lock.acquire()
    test_sum[0] += test_thread_eval[0]
    test_sum[1] += test_thread_eval[1]
    test_sum[2] += test_thread_eval[2]
    test_sum[3] += test_thread_eval[3]
    test_cnt += test_thread_cnt
    lock.release()


lock = threading.Lock()


if __name__ == '__main__':

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset(cmd_args)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start construct graph" % log_time)

    # restore pre-trained u function
    user_model = UserModelLSTM(dataset.f_dim, cmd_args, dataset.max_disp_size)

    user_model.construct_placeholder()
    with tf.variable_scope('model', reuse=False):
        user_model.construct_computation_graph_u()

    saved_path = cmd_args.save_dir+'/'
    saver = tf.train.Saver(max_to_keep=None)
    sess = tf.Session()
    sess.run(tf.variables_initializer(user_model.min_trainable_variables))
    best_save_path = os.path.join(saved_path, 'best-pre1')
    saver.restore(sess, best_save_path)

    # construct policy net
    train_min_opt, train_max_opt, train_loss_min, train_loss_max, train_prec1, train_prec2, train_loss_min_sum, \
    train_loss_max_sum, train_prec1_sum, train_prec2_sum, train_event_cnt = user_model.construct_computation_graph_policy()

    sess.run(tf.initialize_variables(user_model.init_variables))
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, graph completed" % log_time)

    batch_size = 100
    batch = 100

    if cmd_args.dataset == 'lastfm':
        batch_size = 10
        batch = 10

    iterations = cmd_args.num_itrs

    # prepare validation data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)
    vali_thread_user, size_user_vali, max_time_vali, news_cnt_short_vali, u_t_dispid_vali, \
    u_t_dispid_split_ut_vali, u_t_dispid_feature_vali, click_feature_vali, click_sub_index_vali, \
    u_t_clickid_vali, ut_dense_vali = dataset.prepare_validation_data_L2(cmd_args.num_thread, dataset.vali_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare vali data complete" % log_time)

    best_metric = [0.0, 0.0, 0.0, 0.0]

    saver = tf.train.Saver(max_to_keep=None)

    vali_path = cmd_args.save_dir+'/minmax_L2/'
    if not os.path.exists(vali_path):
        os.makedirs(vali_path)

    for i in xrange(iterations):

        training_user = np.random.choice(len(dataset.train_user), batch, replace=False)
        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, start prepare train data" % log_time)

        size_user_tr, max_time_tr, news_cnt_short_tr, u_t_dispid_tr, u_t_dispid_split_ut_tr, \
        u_t_dispid_feature_tr, click_feature_tr, click_sub_index_tr, u_t_clickid_tr, ut_dense_tr = dataset.data_process_for_placeholder_L2(training_user)

        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, prepare train data completed" % log_time)
            print("%s, start first iteration training" % log_time)

        sess.run(train_max_opt, feed_dict={user_model.placeholder['clicked_feature']: click_feature_tr,
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

        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, first iteration training complete" % log_time)

        if np.mod(i, 100) == 0:
            loss_prc = sess.run([train_loss_min, train_loss_max, train_prec1, train_prec2], feed_dict={user_model.placeholder['clicked_feature']: click_feature_tr,
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
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            vali_loss_prc = multithread_compute_vali()
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, training: %.5f, %.5f, %.5f, %.5f, vali: %.5f, %.5f, %.5f, %.5f" %
                  (log_time, i, loss_prc[0], loss_prc[1], loss_prc[2], loss_prc[3], vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2], vali_loss_prc[3]))

            if vali_loss_prc[2] > best_metric[2]:
                best_metric[2] = vali_loss_prc[2]
                best_save_path = os.path.join(vali_path, 'best-pre1')
                best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[3] > best_metric[3]:
                best_metric[3] = vali_loss_prc[3]
                best_save_path = os.path.join(vali_path, 'best-pre2')
                best_save_path = saver.save(sess, best_save_path)
            save_path = os.path.join(vali_path, 'most_recent_iter')
            save_path = saver.save(sess, save_path)

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)
    test_thread_user, size_user_test, max_time_test, news_cnt_short_test, u_t_dispid_test, \
    u_t_dispid_split_ut_test, u_t_dispid_feature_test, click_feature_test, click_sub_index_test, \
    u_t_clickid_test, ut_dense_test = dataset.prepare_validation_data_L2(cmd_args.num_thread, dataset.test_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    best_save_path = os.path.join(vali_path, 'best-pre1')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test()
    vali_loss_prc = multithread_compute_vali()
    print("test!!!best-pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))

    best_save_path = os.path.join(vali_path, 'best-pre2')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test()
    vali_loss_prc = multithread_compute_vali()
    print("test!!!best-pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[3], vali_loss_prc[3]))
