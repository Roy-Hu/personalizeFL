import os
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F
from RL.generic import Trainer
from models import device, models
from models.rl_model import PolicyNetwork, QNetwork
from torch import nn, optim
from torch.distributions import Categorical
from utils.replay_buffer import ReplayBuffer
from utils import normorlize, soft_update


class td3(Trainer):
    def init_params(self):
        super().init_params()

        self.epoch = 0
        self.update_cnt = 0
        self.device_list = None
        self.replay_buffer = ReplayBuffer(self.args.replay_buffer_size)
        self.state = torch.ones(1, self.args.state_dim).to(device)

        self.best_acc= torch.zeros(self.args.num_max_devices) - 1
        self.best_action = torch.zeros((self.args.num_max_devices)).int()
        
        self.action = torch.rand((self.args.num_devices, self.args.action_dim))

        hidden = torch.zeros([1, 1, self.args.FF]).float().to(device)
        cell = torch.zeros([1, 1, self.args.FF]).float().to(device)
        self.h_init = (hidden, cell)
        self.h_out = None

    def init_models(self):
        super().init_models()

        self.critic_1 = QNetwork(self.args).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=self.args.c_lr)

        self.critic_2 = QNetwork(self.args).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=self.args.c_lr)

        self.actor_1 = PolicyNetwork(self.args).to(device)
        self.actor_1_target = copy.deepcopy(self.actor_1)
        self.actor_1_optim = optim.Adam(self.actor_1.parameters(), lr=self.args.a_lr)

        self.actor_2 = PolicyNetwork(self.args).to(device)
        self.actor_2_target = copy.deepcopy(self.actor_2)
        self.actor_2_optim = optim.Adam(self.actor_2.parameters(), lr=self.args.a_lr)

    def softmax_operator(self, q_vals, noise_pdf=None):
        max_q_vals = torch.max(q_vals, 1, keepdim=True).values
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(self.args.beta * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ

        if self.args.imps:
            numerators /= noise_pdf
            denominators /= noise_pdf

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_q_vals = sum_numerators / sum_denominators

        softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals

    def calc_pdf(self, samples, mu=0):
        p_n = self.args.policy_noise
        pdfs = 1 / (p_n * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * p_n**2))
        pdf = torch.prod(pdfs, dim=2).to(device)
        return pdf

    def calculate_reward(self, acc_mean, similarity_rewards):
        sim_rewards = self.args.theta ** similarity_rewards.mean() - 1

        acc_rewards = self.args.theta ** (acc_mean - 0.6) - 1

        rewards = self.args.alpha * sim_rewards + self.args.beta * acc_rewards
        rtn = rewards.clone().to(device)
        self.rewards = rtn
        return rtn
    
    def calculate_best_models(self, rewards_info):
        bipartion_dev_num = int(self.args.num_devices / 2)
        for idx in range(bipartion_dev_num):
            device_idx = rewards_info["devices"][idx]
            act = (rewards_info["actions"][idx][0]).int()
            new_acc = rewards_info["similarity"][idx]
            if self.best_acc[device_idx] < new_acc:
                self.best_acc[device_idx] = new_acc
                self.best_action[device_idx] = act.int()
        print(self.best_action)

    def reset(self):
        self.epoch = 0
        self.update_cnt = 0
        self.replay_buffer = ReplayBuffer(self.args.replay_buffer_size)

    def update(self, update_q1, upd_actor=True, discount=0.99):

        tau = self.args.tau

        h_in, h_out, state, last_action, action, rewards, next_state, done = self.replay_buffer.sample(
            self.args.batch_size
        )

        noise_clip = self.args.noise_clip * self.args.max_action
        policy_noise = self.args.policy_noise * self.args.max_action
        noise = torch.randn((action.shape[0], 1, action.shape[-1]))
        noise = noise * policy_noise
        noise_pdf = self.calc_pdf(noise) if self.args.imps else None

        if update_q1:
            next_action, _ = self.actor_1_target(next_state, action, h_out)
            new_action, _ = self.actor_1(state, last_action, h_in)
        else:
            next_action, _ = self.actor_2_target(next_state, action, h_out)
            new_action, _ = self.actor_2(state, last_action, h_in)

        new_action = new_action.unsqueeze(1)
        next_action = torch.unsqueeze(next_action, 1)
        next_action = next_action + noise.clamp(-noise_clip, noise_clip).to(device)
        next_action = next_action.clamp(-self.args.max_action, self.args.max_action)
        with torch.no_grad():
            next_q1, _ = self.critic_1_target(next_state, next_action, action, h_out)
            next_q2, _ = self.critic_2_target(next_state, next_action, action, h_out)
            next_q = torch.min(next_q1, next_q2)

            next_q = torch.squeeze(next_q, 2)
            next_q  = self.softmax_operator(next_q, noise_pdf)
            target_q = rewards.reshape(-1, 1) +  (1 - done.reshape(-1, 1)) * discount * next_q

        if update_q1:
            q_1, _ = self.critic_1(state, action, last_action, h_in)

            critic_1_loss = F.mse_loss(q_1.reshape(-1, 1).double(), target_q.double()).float()
            self.critic_1_optim.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optim.step()
            self.last_epoch["q_loss1"] = critic_1_loss

            if upd_actor:
                new_q_value, _ = self.critic_1(state, new_action, last_action, h_in)
                actor_1_loss = -new_q_value.mean()
                self.actor_1_optim.zero_grad()
                actor_1_loss.backward()
                self.actor_1_optim.step()

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.actor_1.parameters(), self.actor_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            q_2, _ = self.critic_2(state, action, last_action, h_in)

            critic_2_loss = F.mse_loss(q_2.reshape(-1, 1).double(), target_q.double()).float()
            self.critic_2_optim.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optim.step()
            self.last_epoch["q_loss2"] = critic_2_loss

            if upd_actor:
                new_q_value, _ = self.critic_2(state, new_action, last_action, h_in)
                actor_2_loss = -new_q_value.mean()
                self.actor_2_optim.zero_grad()
                actor_2_loss.backward()
                self.actor_2_optim.step()

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.actor_2.parameters(), self.actor_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, state, next_state, actions, last_actions, rewards_info, h_in=None, h_out=None, done=0):
        self.epoch += 1
        self.last_epoch["done"] = done
        self.last_epoch["action_num"] = self.args.num_devices
        self.last_epoch["reward"] = self.rewards

        acc = []
        update_w = False
        upd_actor = False
        for idx, act in enumerate(rewards_info["actions"]):
            acc_list = rewards_info["acc_list"][idx]
            acc.append(acc_list[int(act[0])])
        bipartion_dev_num = int(self.args.num_devices / 2)
        acc_mean = np.mean(acc[:bipartion_dev_num])
        self.last_epoch["acc"] = acc_mean

        done = torch.tensor(done)
        
        similarity_rewards = rewards_info["similarity"][:bipartion_dev_num]
        reward = self.calculate_reward(acc_mean, similarity_rewards)

        self.replay_buffer.push(
            h_in,
            h_out,
            state.reshape(1, -1).to(device),
            last_actions.reshape(1, -1).to(device),
            actions.reshape(1, -1).to(device),
            reward.reshape(1, -1).to(device),
            next_state.reshape(1, -1).to(device),
            done.reshape(1, -1).to(device),
        )

        if self.epoch > self.args.explore_steps:
            self.update_cnt += 1
            if self.update_cnt % self.args.update_itr == 0:
                self.update_cnt, upd_actor = 0, True

            if self.last_epoch["best_acc"] < acc_mean:
                self.last_epoch["best_acc"] = acc_mean
                update_w = True

            self.calculate_best_models(rewards_info)

            self.update(True, upd_actor)
            self.update(False, upd_actor)

            tb_params = []

            print("|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾RL‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|")

            for key, info in self.last_epoch.items():
                tb_params.append({"title": key, "param": info})
            self.add_tensorboard(tb_params, global_step=self.epoch)

            path = os.path.join(self.tensorboard.file_writer.get_logdir(), "weights.pb")
            self.save_model(self.model.state_dict(), path)

            print(
                "|____________________________________RL____________________________________|"
            )

            return update_w

    def get_current_reward(self):
        return self.rewards.cpu().numpy()

    def get_action(self, state=None, last_action=None, h_in=None, rand_sample=False):
        rtn = torch.zeros((self.args.num_devices, 1))
        softmax = torch.nn.Softmax(dim=-1)

        with torch.no_grad():
            max_action = self.args.max_action
            if self.epoch <= self.args.explore_steps or rand_sample:
                h_out = self.h_init
                sample = torch.randn((self.args.action_dim))
                action = softmax(sample.reshape(self.args.num_devices, -1))
                action = action.reshape(-1)
                if not rand_sample:
                    self.last_epoch["training"] = False
            else:
                state = state.to(device)
                last_action = last_action.to(device)

                action_1, h_out_1 = self.actor_1(state, last_action, h_in)
                action_1 = torch.unsqueeze(action_1, 1)
                action_2, h_out_2 = self.actor_2(state, last_action, h_in)
                action_2 = torch.unsqueeze(action_2, 1)

                q1, _ = self.critic_1(state, action_1, last_action, h_in)
                q2, _ = self.critic_2(state, action_2, last_action, h_in)
                h_out = (h_out_1 if q1 >= q2 else h_out_2)
    
                noise = torch.normal(0.0, max_action * self.args.expl_noise, action_1.shape)
                action = (action_1 if q1 >= q2 else action_2) + noise.to(device)
                action = softmax(action.reshape(self.args.num_devices, self.args.num_models))
                action = action.reshape(-1)
                self.last_epoch["training"] = True

            # action = action.clamp(-max_action, max_action)
            # for idx, act in enumerate(action.reshape(self.args.num_devices, -1)):
            #     index_1 = (self.args.num_models / 2) if act[0] > 0 else 0
            #     index_2 = (self.args.num_models / 4) if act[1] > 0 else 0

            #     index = index_1 + index_2
            #     rtn[idx] = (index + torch.round((act[2] + max_action) / 2)).int()
           
            rtn = action.reshape(self.args.num_devices, self.args.num_models)
            if not rand_sample:
                print('action', rtn[0], rand_sample)
            rtn = torch.argmax(rtn, dim=-1)
            if not rand_sample:
                print('action', rtn)
            rtn = rtn.reshape(-1, 1)
        
        return rtn.cpu(), action.cpu(), h_out

