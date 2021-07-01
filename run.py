import os
import time
import copy
import numpy as np
import torch
import csv
import torchvision as tv
import sys

from RL.td3 import td3
from torch import nn, optim
from models import models, device
from configs.fl_parameter import FLParser
from configs.rl_parameter import RLParser
from utils.dataset import DataLorder
from utils import (
    cifa100_coarse,
    print_epoch_stats,
    set_participating_devices,
    set_seed,
    model_weight_pca,
    normorlize,
    soft_update,
)
from models.mobile import MobileNetV2

# from data import data_distribute
from torch.utils.tensorboard import SummaryWriter

from data import data_distribute

rl_args = RLParser()

round = 1
class Cluster:
    def __init__(self, args, rl_trainer=None):
        set_seed(args.seed)
        self.args = args
        self.new_method = False
        self.rl_trainer = rl_trainer
        self.init_models()
        self.init_dataset()
        self.init_param()

        self.best_model_list = np.zeros(self.args.num_max_devices, dtype=np.int32)
        self.previous_assign = np.ones(args.num_max_devices, dtype=np.int32) * -1
        self.previous_assign_test = np.ones(args.num_max_devices, dtype=np.int32) * -1
        self.accumulate_change = 0
        self.accumulate_change_test = 0
        self.CoPT = rl_args.num_max_devices * rl_args.num_models * 2

    def run(self, rounded, with_rl=True, initial=False):
        args = self.args

        results = []
        actions = None
        update_w = False
        origin_lr = self.args.lr

        data_distribute(self, args.n, self.ds_train_y, True, args.num_max_devices)
        data_distribute(self, args.n_test, self.ds_test_y, False, args.num_max_devices)

        if initial:
            self.setup_initial_devices_models()
            for idx, model in enumerate(self.models):
                torch.save(model.state_dict(),
                            "weights/model_{}".format(
                                                    str(idx)
                                                    + str(self.args.ratio) 
                                                    + '_' 
                                                    + str(self.args.ratio_single)
                                                )
                )
        else:
            for idx, model in enumerate(self.models):
                model.load_state_dict(torch.load(
                            "weights/model_{}".format(
                                                    str(idx)
                                                    + str(self.args.ratio) 
                                                    + '_' 
                                                    + str(self.args.ratio_single)
                                                )
                ))

        ####################################### For RL #######################################
        if with_rl:
            self.epoch = 0
            h_in = self.rl_trainer.h_init
            bipartion_dev_num = int(self.args.num_devices / 2)
            
            device_list = np.array([*range(self.args.num_max_devices)])
            models_weight, _ = self.train(device_list)
            models_weight = models_weight.reshape((device_list.shape[0], -1))
            self.eval(device_list, training=True)
            clust = np.array(self.best_model_list)
        #______________________________________ For RL ______________________________________#

        device_list = set_participating_devices(args.num_max_devices, args.num_devices)

        train = self.eval(device_list, training=True)
        results.append(train)
        test = self.eval(device_list, training=False)
        results.append(test)
        print_epoch_stats(results, self.filters, "FL", "FL")

        ####################################### For RL #######################################
        if with_rl:
            weights = models_weight[device_list].to(device)
            dev_state = torch.tensor(device_list).reshape((-1, 1)).to(device)
            last_actions = torch.tensor(clust[device_list]).reshape((-1, 1)).to(device)
            state = torch.cat((weights, dev_state, last_actions), dim=-1)
            state = state.reshape(-1).unsqueeze(0).unsqueeze(0)

            _, last_actions_prob, _ = self.rl_trainer.get_action(rand_sample=True)
            last_actions_prob = last_actions_prob.unsqueeze(0).unsqueeze(0)

            actions, actions_prob, h_out = self.rl_trainer.get_action(state, last_actions_prob, h_in)

        #______________________________________ For RL ______________________________________#
        
        self.best_model = copy.deepcopy(self.models)

        episode = 0
        for ep in range(args.num_epochs):
            episode += 1
            result = []
            done = 0
            self.epoch = ep

            ####################################### For RL #######################################
            if with_rl:
                self.CoPT += (rl_args.num_devices * 2)
                self.eval(device_list, actions, training=True)
                models_weight, similarity_rewards = self.train(device_list, clust[device_list])
            #______________________________________ For RL ______________________________________#
            else:
                self.CoPT += (self.args.num_devices + self.args.num_devices * self.args.num_models)
                self.train(device_list)
            
            train = self.eval(device_list, actions, training=True)
            test = self.eval(device_list, training=False)
            
            ####################################### For RL #######################################
            if with_rl:
                if episode > rl_args.explore_steps and ep < self.args.step_to_rl_pass_one_m:
                    self.delay_update_models(train, test, update_w)
                if episode % rl_args.episode == 0:
                    done = 1
            #______________________________________ For RL ______________________________________#

            if episode > rl_args.explore_steps:
                if train["acc"] > self.best_train_acc:
                    self.best_train_acc = train["acc"]
                if test["acc"] > self.best_test_acc:
                    self.best_test_acc = test["acc"]

            train["CoPT"] = self.CoPT

            result.append(train)
            result.append(test)

            print_epoch_stats(result, self.filters, "FL", "FL")

            self.args.lr *= 0.999

            ####################################### For RL #######################################
            if with_rl:
                if ep <= self.args.step_to_rl_pass_one_m:
                    weights = models_weight.reshape((self.args.num_devices, self.args.num_models))
                    dev_state = torch.tensor(device_list).reshape((-1, 1)).to(device)
                    last_actions = actions.reshape((-1, 1)).to(device)
                    next_state = torch.cat((weights, dev_state, last_actions), dim=-1)
                    next_state = next_state.reshape(-1).unsqueeze(0).unsqueeze(0)
                    rewards = {
                        "acc_list": train["accuracy"][device_list],
                        "acc_mean": test["acc"],
                        "actions": actions,
                        "devices": device_list,
                        "similarity": similarity_rewards,
                    }
                    update_w = self.rl_trainer.train(
                        state, next_state, actions_prob, last_actions_prob, rewards, h_in, h_out, done)

                    h_in = copy.deepcopy(h_out)
                    last_actions_prob = actions_prob.unsqueeze(0).unsqueeze(0)
                    actions, actions_prob, h_out = self.rl_trainer.get_action(next_state, last_actions_prob, h_in)
                    actions, actions_prob = actions[bipartion_dev_num:], actions_prob[bipartion_dev_num:]

                    n_actions, n_actions_prob, _ = self.rl_trainer.get_action(rand_sample=True)
                    n_actions, n_actions_prob = n_actions[:bipartion_dev_num], n_actions_prob[:bipartion_dev_num]

                    actions = torch.cat((actions, n_actions), dim=0)
                    actions_prob = torch.cat((actions_prob, n_actions_prob), dim=0)

                    state = next_state

                    n_devices_1 = device_list[bipartion_dev_num:]
                    n_devices_2 = []
                    shuffle_idx = np.arange(self.args.num_max_devices)
                    np.random.shuffle(shuffle_idx)
                    for idx in shuffle_idx:
                        if idx not in n_devices_1:
                            n_devices_2.append(idx)
                        if len(n_devices_2) == int(self.args.num_devices / 2):
                            break
                    n_devices_2 = np.array(n_devices_2)
                    device_list = np.concatenate((n_devices_1, n_devices_2))

                    if episode % rl_args.episode == 0:
                        self.CoPT += rl_args.num_max_devices * rl_args.num_models * 2
                        self.args.lr = origin_lr
                        print('Calibrating...')
                        for _ in range(5):
                            device_list = set_participating_devices(self.args.num_max_devices, self.args.num_devices)
                            models_weight, _ = self.train(device_list)
                            models_weight = models_weight.reshape((self.cluster_assign_data_idx_train[d_idx]_list.shape[0], -1))
                            self.eval(device_list, training=True)
                            for d_idx in device_list:
                                clust[d_idx] = self.best_model_list[d_idx]
                            for idx in range(self.args.num_models):
                                soft_update_w = soft_update(self.best_model[idx], self.models[idx], 0.2)
                                self.best_model[idx] = copy.deepcopy(soft_update_w)
                        print('Calibration Finishg...')

                        weights = models_weight.to(device)
                        dev_state = torch.tensor(device_list).reshape((-1, 1)).to(device)
                        last_actions = torch.tensor(clust[device_list]).reshape((-1, 1)).to(device)
                        state = torch.cat((weights, dev_state, last_actions), dim=-1)
                        state = state.reshape(-1).unsqueeze(0).unsqueeze(0)

                        _, last_actions_prob, _ = self.rl_trainer.get_action(rand_sample=True)
                        last_actions_prob = last_actions_prob.unsqueeze(0).unsqueeze(0)

                        actions, actions_prob, h_out = self.rl_trainer.get_action(state, last_actions_prob, h_in)
                else:
                    device_list = set_participating_devices(args.num_max_devices, args.num_devices)
                    actions = None
            #______________________________________ For RL ______________________________________#
            else:
                device_list = set_participating_devices(args.num_max_devices, args.num_devices)

            path = 'log_CoPTLimit_{}_batch_{}_simAlpha_{}_accBeta_{}_theta_{}_lr_{}'.format(
                self.args.CoPT_limit,
                rl_args.batch_size,
                rl_args.alpha,
                rl_args.beta,
                rl_args.theta,
                rl_args.c_lr
            )

            reward = self.rl_trainer.get_current_reward()
            directory = 'exp/' + path + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory + path + '_' + str(rounded) + '.csv', 'a') as csvfile:
                fieldnames = ['CoPT', 'Train_Acc', 'Test_Acc', 'Best_Train_Acc', 'Best_Test_Acc', 'Reward']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'CoPT': self.CoPT, 
                    'Train_Acc': train["acc"],
                    'Test_Acc': test["acc"],
                    'Best_Train_Acc': self.best_train_acc,
                    'Best_Test_Acc': self.best_test_acc,
                    'Reward': reward,
                })

            if self.CoPT > self.args.CoPT_limit:
                break

    def init_dataset(self):
        args = self.args

        ds_train = DataLorder.load(args, training=True)
        self.train_size = DataLorder.size
        ds_train_x, ds_train_y = [], []
        for train in ds_train:
            ds_train_x.append(np.array(train[0]))
            ds_train_y.append(np.array(train[1]))
        self.ds_train_x = torch.tensor(np.array(ds_train_x)).to(device).float()
        self.ds_train_y = torch.tensor(np.array(ds_train_y)).to(device).long()

        ds_test = DataLorder.load(args, training=False)
        self.test_size = DataLorder.size
        ds_test_x, ds_test_y = [], []
        for test in ds_test:
            ds_test_x.append(np.array(test[0]))
            ds_test_y.append(np.array(test[1]))
        self.ds_test_x = torch.tensor(np.array(ds_test_x)).to(device).float()
        self.ds_test_y = torch.tensor(np.array(ds_test_y)).to(device).long()

    def init_param(self):
        self.best_test_acc = 0
        self.best_train_acc = 0
        self.epoch = -1
        self.filters = [
            "time",
            "epoch",
            "is_train",
            "loss",
            "acc",
            "cl_ct_pre",
            "cl_ct_ans",
            "best_acc",
        ]
        self.criterion = nn.CrossEntropyLoss()

        self.accumulate_change = 0

        args = len(sys.argv)

        for i in range(1, args):
            if sys.argv[i] == "--supercls":
                self.args.supercls = True
            elif sys.argv[i] == "--cls_with_supercls":
                self.args.cls_with_supercls == True
            elif sys.argv[i] == "--step_to_pass_one_m":
                self.args.step_to_pass_one_m = int(sys.argv[i + 1])
            elif sys.argv[i] == "--ratio":
                self.args.ratio = []
                r = i + 1
                while r < args:
                    try:
                        float(sys.argv[r])
                    except ValueError:
                        break
                    self.args.ratio.append(float(sys.argv[r]))
                    r += 1
                self.new_method = True
            elif sys.argv[i] == "--ratio_single":
                self.args.ratio_single = []
                r = i + 1
                while r < args:
                    try:
                        float(sys.argv[r])
                    except ValueError:
                        break
                    self.args.ratio_single.append(float(sys.argv[r]))
                    r += 1

        self.writer = SummaryWriter(
            "rl_"
            + str(self.args.ratio)
            + "_"
            + str(self.args.ratio_single)
            + str(self.args.step_to_pass_one_m)
        )
        if not (self.args.cls_with_supercls or self.args.supercls):
            print("run in iid data distribution")

    def init_models(self):
        models = []
        for m_idx in range(self.args.num_models):
            model = MobileNetV2()
            model.classifier = nn.Linear(model.last_channel, self.args.num_cls)
            model = model.to(device)
            models.append(model)

        self.models = np.array(models)

    def delay_update_models(self, train, test, update_w):
        if train["acc"] > self.best_train_acc:
            for idx in range(self.args.num_models):
                soft_update_w = soft_update(self.best_model[idx], self.models[idx], 0.8)
                self.best_model[idx] = copy.deepcopy(soft_update_w)
        elif test["acc"] > self.best_test_acc:
            for idx in range(self.args.num_models):
                soft_update_w = soft_update(self.best_model[idx], self.models[idx], 0.0)
                self.best_model[idx] = copy.deepcopy(soft_update_w)
        elif update_w:
            for idx in range(self.args.num_models):
                soft_update_w = soft_update(self.best_model[idx], self.models[idx], 0.8)
                self.best_model[idx] = copy.deepcopy(soft_update_w)
        elif train["acc"] < self.best_train_acc:
            for idx in range(self.args.num_models):
                soft_update_w = soft_update(self.models[idx], self.best_model[idx], 0.2)
                self.models[idx] = copy.deepcopy(soft_update_w)

    def setup_initial_devices_models(self):
        print("finding good initializer from train data...")
        if self.args.num_devices > 10:
            threshhold = 0.01
        elif self.args.num_devices == 10:
            threshhold = 0.05
        elif self.args.num_devices == 4:
            threshhold = 0.1
        elif self.args.num_devices == 2:
            threshhold = 0.35
        elif self.args.num_devices == 1:
            threshhold = 0.0
        else:
            raise NotImplementedError("only p=1,2,4,10 amd more are supported")

        is_found_good_init = False
        device_list = set_participating_devices(self.args.num_max_devices,int(self.args.num_max_devices * 0.5))
        # device_list = np.array([*range(self.args.num_max_devices)])

        while not is_found_good_init:
            self.init_models()

            since = time.time()
            rtn = self.eval(device_list, training=True)
            rtn["infer_time"] = time.time() - since

            print_epoch_stats([rtn], self.filters, "init", "init")

            cl_ct = rtn["cl_ct_pre"]
            num_nodes = np.sum(cl_ct)

            is_found_good_init = True

            no_device_choosen = 0
            for ct in cl_ct:
                if ct / num_nodes < threshhold:
                    no_device_choosen += 1
                    if no_device_choosen > self.args.init_toleration:
                        is_found_good_init = False
                elif ct / num_nodes > self.args.init_centralize_toleration:
                    is_found_good_init = False

        print("found good initializer...")

    def train(self, device_list, clust=[]):
        def _update(net, target_net, alpha=0.5, beta=0.5):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(alpha * param.data + beta * target_param.data)
            return target_net

        device_models_weight = []
        models = [[] for _ in range(self.args.num_models)]
        for idx, device_idx in enumerate(device_list):
            m_idx = self.best_model_list[device_idx]
            model = copy.deepcopy(self.models[m_idx])
            optimizer = optim.Adam(
                list(model.parameters()), lr=self.args.lr, weight_decay=1e-5
            )

            models[m_idx].append(model)

            inputs = self.ds_train_x[
                np.array(self.cluster_assign_data_idx_train[device_idx])
            ]
            labels = self.ds_train_y[
                np.array(self.cluster_assign_data_idx_train[device_idx])
            ]
            for _ in range(self.args.num_local_epochs):

                logits = model(inputs)

                loss = self.criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                device_model = torch.tensor(model_weight_pca(model, 1))
            device_models_weight.append(device_model)

        for m_idx, model in enumerate(models):
            avg_weight_model = None
            for idx in range(len(model)):
                avg = 1.0 / len(model)
                if idx == 0:
                    avg_weight_model = _update(model[0], model[0], 0.0, avg)
                else:
                    avg_weight_model = _update(model[idx], avg_weight_model, avg, 1.0)
            if avg_weight_model != None:
                self.models[m_idx] = avg_weight_model

        weight_pca = torch.zeros((self.args.num_models, 100))
        for m_idx in range(self.args.num_models):
            with torch.no_grad():
                pca = torch.tensor(model_weight_pca(self.models[m_idx], 1))
                weight_pca[m_idx] = pca.reshape(-1)
        weight_pca = weight_pca.float().to(device)
        
        num_devices = device_list.shape[0]
        sim_rewards = torch.zeros((num_devices))
        cos_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        models_weight = torch.zeros((num_devices, self.args.num_models))
        if len(clust) > 0:
            for idx in range(num_devices):
                c_idx = clust[idx]
                model_w = device_models_weight[idx].reshape(-1).to(device)

                similarity = cos_similarity(model_w, weight_pca)
                models_weight[idx] = ((similarity + 1) / 2).clamp(0.05, 0.95)

                sim_rewards[idx] = ((similarity - 1) / 2).clamp(-0.95, -0.05)[c_idx]
            models_weight = models_weight.reshape(-1).to(device)

        return models_weight, sim_rewards

    def eval(self, device_list, actions=None, training=False):
        args = self.args

        since = time.time()
        if training:
            dataset_x, dataset_y = self.ds_train_x, self.ds_train_y
            cluster_assign_data_idx = self.cluster_assign_data_idx_train
            previous_assign = copy.deepcopy(self.previous_assign)
            cluster_assign_ans_part = self.cluster_assign_ans[device_list]
            folder = "train"
            data_num = args.n
            accumulative_change = self.accumulate_change
            best_acc = self.best_train_acc
        else:
            dataset_x, dataset_y = self.ds_test_x, self.ds_test_y
            cluster_assign_data_idx = self.cluster_assign_data_idx_test
            previous_assign = copy.deepcopy(self.previous_assign_test)
            cluster_assign_ans_part = self.cluster_assign_ans_test[device_list]
            folder = "test"
            data_num = args.n_test
            accumulative_change = self.accumulate_change_test
            best_acc = self.best_test_acc

        acc_list = np.zeros([args.num_max_devices, args.num_models])
        loss_list = np.zeros([args.num_max_devices, args.num_models])

        device_num = args.num_max_devices

        cluster_decision_change = 0
        min_acc_list = []
        min_loss_list = []
        cluster_assign = [-1 for _ in range(device_num)]

        if self.epoch == args.step_to_pass_one_m and training:
            device_list = [*range(device_num)]

        for idx, device_idx in enumerate(device_list):
            data_idx = np.array(cluster_assign_data_idx[device_idx])
            inputs, labels = dataset_x[data_idx], dataset_y[data_idx]
            if actions != None:
                model_list = list(actions[idx])
            elif self.epoch >= args.step_to_pass_one_m and training:
                model_list = [int(self.best_model_list[device_idx])]
            else:
                model_list = [*range(args.num_models)]
            data_idx = np.array(cluster_assign_data_idx[device_idx])
            inputs, labels = dataset_x[data_idx], dataset_y[data_idx]

            for m_idx in range(args.num_models):
                if m_idx not in model_list:
                    loss_list[device_idx][m_idx] = np.inf
                    acc_list[device_idx][m_idx] = -np.inf
                else:
                    with torch.no_grad():
                        logits = self.models[m_idx](inputs)
                        loss = self.criterion(logits, labels)
                        loss_list[device_idx][m_idx] = loss

                        acc = (torch.argmax(logits, dim=-1) == labels).float().sum()
                        acc_list[device_idx][m_idx] = acc / data_num
            min_model_idx = np.argmin(loss_list[device_idx])
            cluster_assign[device_idx] = min_model_idx

            min_acc_list.append(acc_list[device_idx][min_model_idx])
            min_loss_list.append(loss_list[device_idx][min_model_idx])

            if self.epoch < args.step_to_pass_one_m and training:
                self.best_model_list[device_idx] = int(min_model_idx)

            if self.epoch >= 0:
                if (
                    training
                    and self.previous_assign[device_idx] != cluster_assign[device_idx]
                ):
                    cluster_decision_change += 1
                    self.accumulate_change += 1
                    self.previous_assign[device_idx] = cluster_assign[device_idx]
                elif (
                    not training
                    and self.previous_assign_test[device_idx]
                    != cluster_assign[device_idx]
                ):
                    cluster_decision_change += 1
                    self.previous_assign_test[device_idx] = cluster_assign[device_idx]

        accumulative_change += cluster_decision_change

        self.writer.add_scalar(
            "Accumulate_change/{}".format(folder), accumulative_change, self.epoch
        )
        self.writer.add_scalar(
            "Accuracy/{}".format(folder), np.mean(min_acc_list), self.epoch
        )
        self.writer.add_scalar(
            "Loss/{}".format(folder), np.mean(min_loss_list), self.epoch
        )
        self.writer.add_scalar(
            "Cluster_decision_change/{}".format(folder),
            cluster_decision_change,
            self.epoch,
        )
        # self.writer.add_scalar(
        #     'Best accuracy/{}'.format(folder),
        #     best_acc,
        #     self.epoch
        # )

        rtn = {
            "epoch": self.epoch,
            "losses": loss_list,
            "accuracy": acc_list,
            "acc": np.mean(min_acc_list),
            "loss": np.mean(min_loss_list),
            "cl_ct_pre": [
                np.sum(np.array(cluster_assign) == m_idx)
                for m_idx in range(self.args.num_models)
            ],
            "cl_ct_ans": [
                np.sum(np.array(cluster_assign_ans_part) == m_idx)
                for m_idx in range(self.args.num_models)
            ],
            "is_train": training,
            "time": "{} (sec)".format(time.time() - since),
            "best_acc": best_acc,
        }

        return rtn


if __name__ == "__main__":
    # [replay_sample_size, reward(alpha, 1-alpha), reward(base), lr]
    # baseline value is []16, 0.5, 8, 1e-6
    value = [
        # [16, 0.5,  8, 1e-6],
        # [8,  0.5,  8, 1e-6],
        [2,  0.5,  8, 1e-6],
        [32, 0.5,  8, 1e-6],
        [16, 0.0,  8, 1e-6],
        [16, 0.25, 8, 1e-6],
        [16, 0.75, 8, 1e-6],
        [16, 0.5, 32, 1e-6],
        [16, 0.5, 16, 1e-6],
        [16, 0.5,  4, 1e-6],
        [16, 0.5,  4, 1e-4],
        [16, 0.5,  4, 1e-7],
    ]
    for v in value:
        rl_args.batch_size = int(v[0])
        rl_args.replay_buffer_size = int(v[0] * 2)
        rl_args.explore_steps = int((v[0] * 2 + v[0]) / 2)
        rl_args.alpha = v[1]
        rl_args.beta = 1. - v[1]
        rl_args.theta = v[2]
        rl_args.c_lr = v[3]
        rl_args.a_lr = v[3]
        print(
            'bs:', rl_args.batch_size,
            'alpha:', rl_args.alpha,
            'beta:', rl_args.beta,
            'theta:', rl_args.theta,
            'lr:', rl_args.a_lr
        )
        for i in range(3):
            clust = Cluster(FLParser(), td3(rl_args))
            clust.run(i, with_rl=True, initial=False)
