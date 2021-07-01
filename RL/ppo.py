import copy
import random
import torch

import numpy as np

from RL.generic import Trainer
from torch.distributions import Categorical
from models import device, models
from torch import nn, optim


class ppo(Trainer):
    def init_params(self):
        super().init_params()

        self.device_list = None

    def init_models(self):
        super().init_models()

        self.critic = models["Q"](self.args).to(device=device)
        self.target_critic = copy.deepcopy(self.critic).eval()
        self.critic_optim = optim.Adam(list(self.critic.parameters()), lr=self.args.lr)

        self.actor = models[self.args.model](self.args).to(device=device)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.actor_optim = optim.Adam(list(self.actor.parameters()), lr=self.args.lr)

    def combine_action_rewards(self, actions, rewards):

        clamped = torch.clamp(actions, 1 - self.eps, 1 + self.eps)
        weaker_clamp = torch.clamp(actions, 0, self.args.max_clamp)

        return torch.mean(
            torch.max(
                torch.mul(clamped, rewards).mul(-1),
                torch.mul(weaker_clamp, rewards).mul(-1),
            )
        )

    def get_action_probs(self, log_prob, feature, action):

        p = self.ppo_model(feature).detach()
        p = torch.add(self.grid, p / 2.0).unsqueeze(-1)
        c = Categorical(torch.cat((torch.sub(1, p), p), -1))

        return torch.exp(log_prob) / torch.exp(c.log_prob(action))

    def get_features(self, name, dist_features):

        rtn = torch.ones(
            [self.args.num_agents, self.args.num_models], device=device
        ).float()
        for key, value in dist_features.items():
            if value == 1000 or value == -1.0:
                rtn[key[0]][key[1]] = self.last_epoch["{}_feature".format(name)][
                    key[0]
                ][key[1]]
            else:
                rtn[key[0]][key[1]] = torch.tensor(value).float()
        # rtn = rtn / (torch.clamp(rtn.max(0)[0], min=1.0))

        self.last_epoch["{}_feature".format(name)] = rtn

        return rtn

    def calculate_reward(self):

        R_acc = self.args.alpha ** (self.args.FL_avg - self.args.baseline) - 1

        num = (torch.sum(self.action) - self.args.num_agents) / (
            self.total_models - self.args.num_agents
        )

        growth_rate = self.calculate_growth_rate()

        diff = torch.tensor(self.last_epoch["best_acc"] - self.args.FL_avg).float()
        ratio = (torch.log2(torch.tensor(self.epoch + 1).float()) / 2.0).to(device)
        if growth_rate > 0.0:
            if self.last_epoch["best_acc"] < self.args.FL_avg:
                R_num = (self.args.beta / ratio) ** -num - 1
            else:
                R_num = (self.args.beta * ratio) ** -num - 1
        else:
            beta = torch.clamp(diff, (self.args.beta / ratio), self.args.beta)
            R_num = beta.to(device) ** -(1 - num) - 1

        reward = self.args.theta * R_acc + (1 - self.args.theta) * R_num
        self.last_epoch["reward"] = reward

        print(" R_num: ", R_num, " R_acc: ", R_acc)
        self.last_epoch["growth_rate"] = growth_rate
        if self.last_epoch["best_acc"] < self.args.FL_avg:
            self.last_epoch["best_acc"] = self.args.FL_avg

        return reward

    def calculate_growth_rate(self):

        if self.last_epoch["acc"] != 0.0:
            growth_rate = (self.args.FL_avg / self.last_epoch["acc"]) ** (
                1 / self.args.training_step
            ) - 1
        else:
            growth_rate = 0.0
        return growth_rate

    def run(self):

        loss = 0
        training = False

        acc_features = self.get_features("acc", self.args.FL_acc)
        loss_features = self.get_features("loss", self.args.FL_loss)

        features = torch.cat((loss_features, acc_features), -1)
        features = features / (torch.clamp(features.max(0)[0], min=1.0))
        self.last_epoch["feature"] = features

        if (self.args.FL_step + 1) % self.args.training_step == 0:
            print("|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾RL‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|")

            training = True
            self.last_epoch["rand"] = random.random() < self.eps

            self.ppo_model = self.ppo_model.train()

            prediction = self.ppo_model(features)
            self.last_epoch["predict_samples"] = prediction[0:5]

            if self.last_epoch["rand"]:
                p = torch.sub(self.grid, prediction / 2.0).unsqueeze(-1)
            else:
                p = torch.add(self.grid, prediction / 2.0).unsqueeze(-1)

            dist = Categorical(torch.cat((torch.sub(1, p), p), -1))

            self.action = dist.sample()

            ### Add the action probability to history ###
            prob = (
                self.get_action_probs(
                    dist.log_prob(self.action.to(device)),
                    features,
                    self.action.to(device),
                )
                .view(-1)
                .unsqueeze(0)
            )
            # prob = dist.log_prob(self.action.to(device)).view(-1).unsqueeze(0)

            self.history = torch.cat((self.history, prob))

            reward = self.calculate_reward()
            self.costs.append(reward)

            ### Calculates the episodic and greedy rewards at each point in the last episode ###
            Rg = 0
            discounted = []
            for r in range(len(self.costs) - 1, -1, -1):
                Rg = self.costs[r] + self.args.gamma * Rg
                discounted.insert(0, Rg)

            discounted_rewards = (
                torch.tensor(discounted, device=device).float().unsqueeze(-1)
            )
            self.last_epoch["avg_reward"] = (
                torch.mean(discounted_rewards).detach().clone().float()
            )

            loss = self.combine_action_rewards(self.history, discounted_rewards)
            # loss = torch.mean((self.history * discounted_rewards).mul(-1))

            self.last_epoch["acc"] = self.args.FL_avg
            self.epoch += 1

        self.last_epoch["action_num"] = torch.sum(self.action)

        return training, loss, self.last_epoch

    def get_action(self, device_list):
        action_probs = self.actor(self.state[device_list])
        dist = Categorical(action_probs)

        self.action = dist.sample()

        action_num = self.args.topk * self.args.num_devices

        self.action_num.append(action_num)
        self.last_epoch["action_num"] = action_num
        self.last_epoch["action_avg"] = np.mean(self.action_num)
        return self.action