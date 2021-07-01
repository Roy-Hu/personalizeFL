import copy
import torch
import random
import numpy as np
from models import device

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, h_in, h_out, state, last_action, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (h_in, h_out, state, last_action, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        h_in, c_in, h_out, c_out, state, last_action, action, reward, next_state, done = \
            [], [], [], [], [], [], [], [], [], []
        for sample in batch:
            (hi, ci), (ho, co), s, la, a, r, ns, d = sample
            h_in.append(hi)
            c_in.append(ci)
            h_out.append(ho)
            c_out.append(co)
            state.append(s.cpu().numpy())
            last_action.append(la.cpu().numpy())
            action.append(a.cpu().numpy())
            reward.append(r)
            next_state.append(ns.cpu().numpy())
            done.append(d)

        h_in = torch.cat(h_in, dim=-2).detach().to(device)
        h_out = torch.cat(h_out, dim=-2).detach().to(device)
        c_in = torch.cat(c_in, dim=-2).detach().to(device)
        c_out = torch.cat(c_out, dim=-2).detach().to(device)
        state = torch.tensor(state).to(device)
        last_action = torch.tensor(last_action).to(device)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        next_state = torch.tensor(next_state).to(device)
        done = torch.tensor(done).to(device)

        return (h_in, c_in), (h_out, c_out), state, last_action, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)