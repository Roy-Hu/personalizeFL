import copy
import numpy as np
import torch
import random
from timeit import default_timer as timer

from RL.td3 import td3
from torch import nn, optim
from models import models, device
from configs.fl_parameter import FLParser
from utils.dataset import DataLorder
from utils import (
    set_participating_devices,
    set_seed,
    get_params_grad,normalization
)
from models.convnet import Net
from data_random import data_distribute
# from data_distribution import data_distribute

import matplotlib.pyplot as plt

class personalizeFL:
    def __init__(self, args, rl_trainer=None):
        set_seed(args.seed)
        self.args = args
        self.init_models()
        self.init_dataset()
        self.weights = np.identity(self.args.num_max_devices)
        self.beta = [ [] for _ in  range(self.args.num_max_devices)]
        self.var = [ [] for _ in  range(self.args.num_max_devices)]

    def compute_variance(self,pre_model,inputs,labels):
        model = copy.deepcopy(pre_model)
        v = [torch.randn(p.size()).to(device) for p in model.parameters()]  
        v = normalization(v)  
        var = 0.

        l_hessians = []
        for idx,(img, label) in enumerate(zip(inputs,labels)):
            model.zero_grad()
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)

            output = model(img)

            loss = self.criterion(output, label)
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(model)

            l_Hv = torch.autograd.grad(gradsH,
                    params,
                    grad_outputs=v,
                    only_inputs=True,
                    retain_graph=False)

            for i,m_i in enumerate(l_Hv):
                if i == 0:
                    l_hessian = torch.flatten(m_i)
                else:
                    l_hessian = torch.cat((l_hessian,torch.flatten(m_i)),-1)

            if idx == 0:
                cost_hessian = l_hessian   
            else:
                cost_hessian += l_hessian 

            l_hessians.append(l_hessian)

        cost_hessian /= self.args.n_test

        for i,loss_hessian in enumerate(l_hessians):
            dist = cost_hessian-loss_hessian

            var += torch.linalg.norm(dist)

        var = (var * self.args.lr) / self.args.n_test
        model.zero_grad()

        return var

    def init_models(self):
        models = []

        for _ in range(self.args.num_max_devices):
            model = Net()
            model = model.to(device)
            models.append(model)

        self.models = np.array(models)
        self.criterion = nn.CrossEntropyLoss()

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

    def RL(self,d_idx):
        device_list = [d_idx]
        matrix_row_d_idx = []

        for weight in self.weights[d_idx]:
            matrix_row_d_idx.append(weight)

        choosen = np.zeros(len(self.weights[d_idx]))
        choosen = np.array(choosen, dtype=bool)
        choosen[d_idx] = True
        matrix_row_d_idx[d_idx] = -np.inf

        for _ in range(self.args.num_download):
            idx = -1
            if self.args.epsilon > np.random.rand(1):
                idx = random.randint(0,self.args.num_max_devices - 1)
                while choosen[idx]:
                    idx = random.randint(0,self.args.num_max_devices - 1)
            else:
                max_idx = []
                for i in range(self.args.num_max_devices):
                    if matrix_row_d_idx[i] == max(matrix_row_d_idx):
                        if choosen[i]:
                            continue
                        max_idx.append(i)
                idx = random.choice(max_idx)
                                    
            device_list.append(idx)
            matrix_row_d_idx[idx] = -np.inf
            choosen[idx] = True

            return device_list
    def compute_weight(self,d_idx,inputs,labels):
        pre_model = Net()
        checkpoint = torch.load("preModels.pth")
        pre_model.load_state_dict(checkpoint["model_{}".format(d_idx)])  # Choose whatever GPU device number you want
        pre_model.to(device)

        # self.var[d_idx].append(self.compute_variance(pre_model,inputs,labels))

        logits_i = pre_model(inputs)
        loss_i = self.criterion(logits_i, labels)
        device_list = self.RL(d_idx)

        weights = []
        for device_idx in device_list:
            logits_n = self.models[device_idx](inputs)
            loss_n = self.criterion(logits_n, labels)

            for idx,(m_i,m_n) in enumerate(zip(pre_model.parameters(),self.models[device_idx].parameters())):
                if idx == 0:
                    dist = torch.flatten(m_n-m_i)
                else:
                    dist = torch.cat((dist,torch.flatten(m_n-m_i)),-1)

            dist = torch.linalg.norm(dist)

            if dist == 0 :
                weight = self.weights[d_idx][device_idx] / self.args.cur_epoch
            else:
                # weight = ((loss_i - loss_n) + (self.beta[device_idx][-1] + 0.5*self.var[d_idx][-1])) / dist
                weight = ((loss_i - loss_n) + (self.beta[device_idx][-1] + self.args.hyper_var)) / dist
                # weight = (loss_i - loss_n) / dist
                
            weights.append(weight)
            self.weights[d_idx][device_idx] += weight

        return weights,device_list

    def normorlize_weight(self,weights):
        total = 0
        norm_weight = []

        for idx,weight in enumerate(weights):
            weights[idx] = max(weight,0)
            total += weights[idx]
        
        if total == 0:
            total = 1
            weights[0] = 1

        for weight in weights:
            norm_weight.append(weight/total)
        
        return norm_weight

    def download_model(self,device_list):
        def _update(net, target_net, alpha=0.5, beta=0.5):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(alpha * param.data + beta * target_param.data)
            return target_net

        avg_weight_models = []
        for d_idx in device_list:
            inputs = self.ds_test_x[
                np.array(self.test_data_idx[d_idx])
            ]
            labels = self.ds_test_y[
                np.array(self.test_data_idx[d_idx])
            ]

            weights,top_n_device = self.compute_weight(d_idx,inputs,labels)
            
            norm_weights = self.normorlize_weight(weights)
        
            first_nonezero = False
            for idx, device_idx in enumerate(top_n_device):
                if not first_nonezero and norm_weights[idx] != 0:
                    avg_weight_model = copy.deepcopy(self.models[device_idx])
                    avg_weight_model = _update(self.models[device_idx], avg_weight_model, 0.0, norm_weights[idx])
                    first_nonezero = True
                elif  first_nonezero and norm_weights[idx] != 0:
                    avg_weight_model = _update(self.models[device_idx], avg_weight_model, norm_weights[idx], 1.0)
                        
            avg_weight_models.append(avg_weight_model)

        for i,d_idx in enumerate(device_list):
            self.models[d_idx] = avg_weight_models[i]
        
    def train(self,device_list):
        accuracy = 0
        losses = 0
        premodel = {}

        for device_idx in device_list:
            model = copy.deepcopy(self.models[device_idx])
            
            optimizer = optim.Adam(
                list(model.parameters()), lr=self.args.lr , weight_decay=1e-5
            )
            inputs = self.ds_train_x[
                np.array(self.train_data_idx[device_idx])
            ]
            labels = self.ds_train_y[
                np.array(self.train_data_idx[device_idx])
            ]

            for epoch in range(self.args.num_local_epochs):
                optimizer.zero_grad()
                logits = model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                for idx,param in enumerate(model.parameters()):
                    if idx == 0:
                        gradient = torch.flatten(param.grad)
                    else:
                        gradient = torch.cat((gradient,torch.flatten(param.grad)),-1)
        
                if epoch ==0:
                    gradients = gradient
                else:
                    gradients += gradient
                
                optimizer.step()
                if epoch == self.args.num_local_epochs-1:
                    accuracy += torch.sum(labels == logits.argmax(dim=-1)) / self.args.n
                    losses += loss 

                optimizer.zero_grad()

            premodel["model_{}".format(device_idx)] = self.models[device_idx].state_dict()
            self.models[device_idx] = model       
            self.beta[device_idx].append(self.args.lr*torch.linalg.norm(gradients/5.))
        
        checkpoint = torch.load("preModels.pth")

        for idx in range(self.args.num_max_devices):
            if idx not in device_list:
                premodel["model_{}".format(idx)] = checkpoint["model_{}".format(idx)]
                

        torch.save(premodel,"preModels.pth")
        losses /= len(device_list)
        accuracy /= len(device_list)

        return float(accuracy),float(losses)
    def test(self):
        accuracy = 0
        losses = 0
        device_list = np.arange(self.args.num_max_devices)
        for device_idx in device_list:
            inputs = self.ds_test_x[
                np.array(self.test_data_idx[device_idx])
            ]
            labels = self.ds_test_y[
                np.array(self.test_data_idx[device_idx])
            ]
            
            logits = self.models[device_idx](inputs)
            loss = self.criterion(logits, labels)

            accuracy += torch.sum(labels == logits.argmax(dim=-1)) / self.args.n_test
            losses += loss 
        
        losses /= len(device_list)
        accuracy /= len(device_list)

        return float(accuracy),float(losses)

    def run(self):
        accuracy = []
        losses = []

        data_distribute(self, self.args.n, self.ds_train_y, True, self.args.num_max_devices)
        data_distribute(self, self.args.n_test, self.ds_test_y, False, self.args.num_max_devices)

        args = self.args

        device_list = np.arange(args.num_max_devices)
        acc,loss = self.train(device_list)
        print("Training Epoch ",0,": acc = ","%.4f" % acc,", loss = ","%.4f" % loss)
        acc,loss = self.test()
        accuracy.append(acc)
        losses.append(loss)
        print("Testing  Epoch ",0,": acc = ","%.4f" % acc,", loss = ","%.4f" % loss)

        for epoch in range(args.num_epochs):
            self.args.cur_epoch = epoch 
            device_list = set_participating_devices(args.num_max_devices, args.num_devices)
            self.download_model(device_list)
            acc,loss = self.train(device_list)
            print("Training Epoch ",epoch+1,": acc = ","%.4f" % acc,", loss = ","%.4f" % loss)
            acc,loss = self.test()
            print("Testing  Epoch ",epoch+1,": acc = ","%.4f" % acc,", loss = ","%.4f" % loss)
            accuracy.append(acc)
            losses.append(loss)

            if (epoch+1) % 5 == 0:
                plt.imshow(self.weights,interpolation='none')
                plt.savefig("weights.png")
            
        plt.close()
        plt.plot(self.beta[0])
        plt.savefig('gradient.png')

        plt.close()

        plt.plot(self.var[0])
        plt.savefig('var.png')

        plt.close()
        plt.plot(accuracy)
        plt.savefig('acc.png')

        plt.close()
        plt.plot(losses)
        plt.savefig('loss.png')

if __name__ == "__main__":

    pFL = personalizeFL(FLParser())
    pFL.run()
