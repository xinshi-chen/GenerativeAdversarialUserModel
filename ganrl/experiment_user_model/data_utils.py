import numpy as np
import pickle
import datetime
import itertools
import os


class Dataset(object):

    def __init__(self, args):
        self.data_folder = args.data_folder
        self.dataset = args.dataset
        self.model_type = args.user_model
        self.band_size = args.pw_band_size

        data_filename = os.path.join(args.data_folder, args.dataset+'.pkl')
        f = open(data_filename, 'rb')
        data_behavior = pickle.load(f)
        item_feature = pickle.load(f)
        f.close()
        # data_behavior[user][0] is user_id
        # data_behavior[user][1][t] is displayed list at time t
        # data_behavior[user][2][t] is picked id at time t
        self.size_item = len(item_feature)
        self.size_user = len(data_behavior)
        self.f_dim = len(item_feature[0])

        # Load user splits
        filename = os.path.join(self.data_folder, self.dataset+'-split.pkl')
        pkl_file = open(filename, 'rb')
        self.train_user = pickle.load(pkl_file)
        self.vali_user = pickle.load(pkl_file)
        self.test_user = pickle.load(pkl_file)
        pkl_file.close()

        # Process data

        k_max = 0
        for d_b in data_behavior:
            for disp in d_b[1]:
                k_max = max(k_max, len(disp))

        self.data_click = [[] for x in xrange(self.size_user)]
        self.data_disp = [[] for x in xrange(self.size_user)]
        self.data_time = np.zeros(self.size_user, dtype=np.int)
        self.data_news_cnt = np.zeros(self.size_user, dtype=np.int)
        self.feature = [[] for x in xrange(self.size_user)]
        self.feature_click = [[] for x in xrange(self.size_user)]

        for user in xrange(self.size_user):
            # (1) count number of clicks
            click_t = 0
            num_events = len(data_behavior[user][1])
            click_t += num_events
            self.data_time[user] = click_t
            # (2)
            news_dict = {}
            self.feature_click[user] = np.zeros([click_t, self.f_dim])
            click_t = 0
            for event in xrange(num_events):
                disp_list = data_behavior[user][1][event]
                pick_id = data_behavior[user][2][event]
                for id in disp_list:
                    if id not in news_dict:
                        news_dict[id] = len(news_dict)  # for each user, news id start from 0
                id = pick_id
                self.data_click[user].append([click_t, news_dict[id]])
                self.feature_click[user][click_t] = item_feature[id]
                for idd in disp_list:
                    self.data_disp[user].append([click_t, news_dict[idd]])
                click_t += 1  # splitter a event with 2 clickings to 2 events

            self.data_news_cnt[user] = len(news_dict)

            self.feature[user] = np.zeros([self.data_news_cnt[user], self.f_dim])

            for id in news_dict:
                self.feature[user][news_dict[id]] = item_feature[id]
            self.feature[user] = self.feature[user].tolist()
            self.feature_click[user] = self.feature_click[user].tolist()
        self.max_disp_size = k_max

    def random_split_user(self):
        num_users = len(self.train_user) + len(self.vali_user) + len(self.test_user)
        shuffle_order = np.arange(num_users)
        np.random.shuffle(shuffle_order)
        self.train_user = shuffle_order[0:len(self.train_user)].tolist()
        self.vali_user = shuffle_order[len(self.train_user):len(self.train_user)+len(self.vali_user)].tolist()
        self.test_user = shuffle_order[len(self.train_user)+len(self.vali_user):].tolist()

    def data_process_for_placeholder(self, user_set):

        if self.model_type == 'PW':
            sec_cnt_x = 0
            news_cnt_short_x = 0
            news_cnt_x = 0
            click_2d_x = []
            disp_2d_x = []

            tril_indice = []
            tril_value_indice = []

            disp_2d_split_sec = []
            feature_clicked_x = []

            disp_current_feature_x = []
            click_sub_index_2d = []

            for u in user_set:
                t_indice = []
                for kk in xrange(min(self.band_size-1, self.data_time[u]-1)):
                    t_indice += map(lambda x: [x + kk+1 + sec_cnt_x, x + sec_cnt_x], np.arange(self.data_time[u] - (kk+1)))

                tril_indice += t_indice
                tril_value_indice += map(lambda x: (x[0] - x[1] - 1), t_indice)

                click_2d_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], self.data_click[u])
                click_2d_x += click_2d_tmp

                disp_2d_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], self.data_disp[u])
                click_sub_index_tmp = map(lambda x: disp_2d_tmp.index(x), click_2d_tmp)

                click_sub_index_2d += map(lambda x: x+len(disp_2d_x), click_sub_index_tmp)
                disp_2d_x += disp_2d_tmp
                disp_2d_split_sec += map(lambda x: x[0] + sec_cnt_x, self.data_disp[u])

                sec_cnt_x += self.data_time[u]
                news_cnt_short_x = max(news_cnt_short_x, self.data_news_cnt[u])
                news_cnt_x += self.data_news_cnt[u]
                disp_current_feature_x += map(lambda x: self.feature[u][x], [idd[1] for idd in self.data_disp[u]])
                feature_clicked_x += self.feature_click[u]

            return click_2d_x, disp_2d_x, \
                   disp_current_feature_x, sec_cnt_x, tril_indice, tril_value_indice, \
                   disp_2d_split_sec, news_cnt_short_x, click_sub_index_2d, feature_clicked_x

        else:
            news_cnt_short_x = 0
            u_t_dispid = []
            u_t_dispid_split_ut = []
            u_t_dispid_feature = []

            u_t_clickid = []

            size_user = len(user_set)
            max_time = 0

            click_sub_index = []

            for u in user_set:
                max_time = max(max_time, self.data_time[u])

            user_time_dense = np.zeros([size_user, max_time], dtype=np.float32)
            click_feature = np.zeros([max_time, size_user, self.f_dim])

            for u_idx in xrange(size_user):
                u = user_set[u_idx]

                u_t_clickid_tmp = []
                u_t_dispid_tmp = []

                for x in self.data_click[u]:
                    t, click_id = x
                    click_feature[t][u_idx] = self.feature[u][click_id]
                    u_t_clickid_tmp.append([u_idx, t, click_id])
                    user_time_dense[u_idx, t] = 1.0

                u_t_clickid = u_t_clickid + u_t_clickid_tmp

                for x in self.data_disp[u]:
                    t, disp_id = x
                    u_t_dispid_tmp.append([u_idx, t, disp_id])
                    u_t_dispid_split_ut.append([u_idx, t])
                    u_t_dispid_feature.append(self.feature[u][disp_id])

                click_sub_index_tmp = map(lambda x: u_t_dispid_tmp.index(x), u_t_clickid_tmp)
                click_sub_index += map(lambda x: x+len(u_t_dispid), click_sub_index_tmp)

                u_t_dispid = u_t_dispid + u_t_dispid_tmp
                news_cnt_short_x = max(news_cnt_short_x, self.data_news_cnt[u])

            if self.model_type != 'LSTM':
                print('model type not supported. using LSTM')

            return size_user, max_time, news_cnt_short_x, u_t_dispid, u_t_dispid_split_ut, np.array(u_t_dispid_feature),\
                   click_feature, click_sub_index, u_t_clickid, user_time_dense

    def data_process_for_placeholder_L2(self, user_set):
        news_cnt_short_x = 0
        u_t_dispid = []
        u_t_dispid_split_ut = []
        u_t_dispid_feature = []

        u_t_clickid = []

        size_user = len(user_set)
        max_time = 0

        click_sub_index = []

        for u in user_set:
            max_time = max(max_time, self.data_time[u])

        user_time_dense = np.zeros([size_user, max_time], dtype=np.float32)
        click_feature = np.zeros([max_time, size_user, self.f_dim])

        for u_idx in xrange(size_user):
            u = user_set[u_idx]

            item_cnt = [{} for _ in xrange(self.data_time[u])]

            u_t_clickid_tmp = []
            u_t_dispid_tmp = []
            for x in self.data_disp[u]:
                t, disp_id = x
                u_t_dispid_split_ut.append([u_idx, t])
                u_t_dispid_feature.append(self.feature[u][disp_id])
                if disp_id not in item_cnt[t]:
                    item_cnt[t][disp_id] = len(item_cnt[t])
                u_t_dispid_tmp.append([u_idx, t, item_cnt[t][disp_id]])

            for x in self.data_click[u]:
                t, click_id = x
                click_feature[t][u_idx] = self.feature[u][click_id]
                u_t_clickid_tmp.append([u_idx, t, item_cnt[t][click_id]])
                user_time_dense[u_idx, t] = 1.0

            u_t_clickid = u_t_clickid + u_t_clickid_tmp

            click_sub_index_tmp = map(lambda x: u_t_dispid_tmp.index(x), u_t_clickid_tmp)
            click_sub_index += map(lambda x: x+len(u_t_dispid), click_sub_index_tmp)

            u_t_dispid = u_t_dispid + u_t_dispid_tmp
            # news_cnt_short_x = max(news_cnt_short_x, data_news_cnt[u])
            news_cnt_short_x = self.max_disp_size

        return size_user, max_time, news_cnt_short_x, \
               u_t_dispid, u_t_dispid_split_ut, np.array(u_t_dispid_feature), click_feature, click_sub_index, \
               u_t_clickid, user_time_dense

    def prepare_validation_data_L2(self, num_sets, v_user):
        vali_thread_u = [[] for _ in xrange(num_sets)]
        size_user_v = [[] for _ in xrange(num_sets)]
        max_time_v = [[] for _ in xrange(num_sets)]
        news_cnt_short_v = [[] for _ in xrange(num_sets)]
        u_t_dispid_v = [[] for _ in xrange(num_sets)]
        u_t_dispid_split_ut_v = [[] for _ in xrange(num_sets)]
        u_t_dispid_feature_v = [[] for _ in xrange(num_sets)]
        click_feature_v = [[] for _ in xrange(num_sets)]
        click_sub_index_v = [[] for _ in xrange(num_sets)]
        u_t_clickid_v = [[] for _ in xrange(num_sets)]
        ut_dense_v = [[] for _ in xrange(num_sets)]
        for ii in xrange(len(v_user)):
            vali_thread_u[ii % num_sets].append(v_user[ii])
        for ii in xrange(num_sets):
            size_user_v[ii], max_time_v[ii], news_cnt_short_v[ii], u_t_dispid_v[ii],\
            u_t_dispid_split_ut_v[ii], u_t_dispid_feature_v[ii], click_feature_v[ii], \
            click_sub_index_v[ii], u_t_clickid_v[ii], ut_dense_v[ii] = self.data_process_for_placeholder_L2(vali_thread_u[ii])
        return vali_thread_u, size_user_v, max_time_v, news_cnt_short_v, u_t_dispid_v, u_t_dispid_split_ut_v,\
               u_t_dispid_feature_v, click_feature_v, click_sub_index_v, u_t_clickid_v, ut_dense_v

    def prepare_validation_data(self, num_sets, v_user):

        if self.model_type == 'PW':
            vali_thread_u = [[] for _ in xrange(num_sets)]
            click_2d_v = [[] for _ in xrange(num_sets)]
            disp_2d_v = [[] for _ in xrange(num_sets)]
            feature_v = [[] for _ in xrange(num_sets)]
            sec_cnt_v = [[] for _ in xrange(num_sets)]
            tril_ind_v = [[] for _ in xrange(num_sets)]
            tril_value_ind_v = [[] for _ in xrange(num_sets)]
            disp_2d_split_sec_v = [[] for _ in xrange(num_sets)]
            feature_clicked_v = [[] for _ in xrange(num_sets)]
            news_cnt_short_v = [[] for _ in xrange(num_sets)]
            click_sub_index_2d_v = [[] for _ in xrange(num_sets)]
            for ii in xrange(len(v_user)):
                vali_thread_u[ii % num_sets].append(v_user[ii])
            for ii in xrange(num_sets):
                click_2d_v[ii], disp_2d_v[ii], feature_v[ii], sec_cnt_v[ii], tril_ind_v[ii], tril_value_ind_v[ii], \
                disp_2d_split_sec_v[ii], news_cnt_short_v[ii], click_sub_index_2d_v[ii], feature_clicked_v[ii] = self.data_process_for_placeholder(vali_thread_u[ii])
            return vali_thread_u, click_2d_v, disp_2d_v, feature_v, sec_cnt_v, tril_ind_v, tril_value_ind_v, \
                   disp_2d_split_sec_v, news_cnt_short_v, click_sub_index_2d_v, feature_clicked_v

        else:
            if self.model_type != 'LSTM':
                print('model type not supported. using LSTM')
            vali_thread_u = [[] for _ in xrange(num_sets)]
            size_user_v = [[] for _ in xrange(num_sets)]
            max_time_v = [[] for _ in xrange(num_sets)]
            news_cnt_short_v = [[] for _ in xrange(num_sets)]
            u_t_dispid_v = [[] for _ in xrange(num_sets)]
            u_t_dispid_split_ut_v = [[] for _ in xrange(num_sets)]
            u_t_dispid_feature_v = [[] for _ in xrange(num_sets)]
            click_feature_v = [[] for _ in xrange(num_sets)]
            click_sub_index_v = [[] for _ in xrange(num_sets)]
            u_t_clickid_v = [[] for _ in xrange(num_sets)]
            ut_dense_v = [[] for _ in xrange(num_sets)]
            for ii in xrange(len(v_user)):
                vali_thread_u[ii % num_sets].append(v_user[ii])
            for ii in xrange(num_sets):
                size_user_v[ii], max_time_v[ii], news_cnt_short_v[ii], u_t_dispid_v[ii],\
                u_t_dispid_split_ut_v[ii], u_t_dispid_feature_v[ii], click_feature_v[ii], \
                click_sub_index_v[ii], u_t_clickid_v[ii], ut_dense_v[ii] = self.data_process_for_placeholder(vali_thread_u[ii])
            return vali_thread_u, size_user_v, max_time_v, news_cnt_short_v, u_t_dispid_v, u_t_dispid_split_ut_v,\
                   u_t_dispid_feature_v, click_feature_v, click_sub_index_v, u_t_clickid_v, ut_dense_v
