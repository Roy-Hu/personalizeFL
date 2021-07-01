import os
import copy
import torch
import tensorboardX

import numpy as np
import datetime as dt

from models import device, models
from torch import nn, optim
from time import time
from utils import normorlize


class Trainer:
    def __init__(self, args):

        self.args = args

        self.init_models()
        self.init_params()
        self.init_tensorboard()

        if self.args.load:
            self.load_model(self.model, self.args.load)

    def init_tensorboard(self):

        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = tensorboardX.SummaryWriter(
            log_dir="{}/log_train_step-{}_eps-{}_alpha-{}_beta-{}_theta-{}".format(
                self.args.log_dir,
                self.args.training_step,
                self.args.init_eps,
                self.args.alpha,
                self.args.beta,
                self.args.theta,
            )
        )

    def init_params(self):

        self.max_rate = -np.inf
        self.min_cost = np.inf

        self.costs = []
        self.action_num = []
        self.total_models = self.args.num_devices * self.args.topk
        self.rewards = torch.ones(self.args.num_max_devices)
        for idx in range(self.args.num_max_devices):
            self.rewards[idx] = -1

        self.last_epoch = {
            "acc": 0.0,
            "done": 0,
            "reward": -1.0,
            "growth_rate": 0.0,
            "sample": None,
            "action_num": 0,
            "action_avg": 0,
            "best_acc": 0.0,
            "acc_list": None,
            "loss_list": None,
            "q_loss1": 0.0,
            "q_loss2": 0.0,
            "training": False,
        }

    def init_models(self):

        self.model = models[self.args.model](self.args).to(device=device)

    def save_model(self, model_dict, path):

        torch.save(model_dict, path)

    def add_tensorboard(self, params, global_step):

        print_filters = [
            "predict_samples",
        ]
        add_filters = [
            "acc",
            "done",
            "best_acc",
            "learning",
            "reward",
            "q_loss1",
            "q_loss2",
            "training",
        ]
        filters = print_filters + add_filters
        for param in params:
            if param["title"] in filters:
                print(" {}: {}".format(param["title"], param["param"]))
            if param["title"] in add_filters:
                self.tensorboard.add_scalar(
                    param["title"], param["param"], global_step=global_step
                )
