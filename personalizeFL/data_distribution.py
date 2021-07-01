import numpy as np
import time
from math import floor

def data_distribute(self, num_data, dataset, training, device_num):
    args = self.args
    datas = []

    if args.dataset == 'CIFA10':
        cls_per_distribution = int(args.num_cls / args.distribution)
        distribution = [[] for _ in range(args.distribution)]

    for d_idx, c_idx in enumerate(dataset):
        if args.dataset == 'CIFA10':
            distribution_idx = floor(c_idx / cls_per_distribution)
            distribution[distribution_idx].append(d_idx)

    for idx in range(device_num):
        if args.dataset == 'CIFA10':
            dis = floor(idx / (device_num/args.distribution))
            tmp = np.random.choice(distribution[dis], int(num_data),-1)
                    
        datas.append(tmp)
        
    if training:
        self.train_data_idx = np.array(datas)
    else:
        self.test_data_idx = np.array(datas)
