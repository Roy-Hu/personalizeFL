import numpy as np

def data_distribute(self, num_data, dataset, training, device_num):
    np.random.seed(0)
    args = self.args
    datas = []

    if args.dataset == 'CIFA10':
        cls_per_device = int(args.num_cls / args.distribution)
        class_data = [[] for _ in range(args.num_cls)]

    for d_idx, c_idx in enumerate(dataset):
        if args.dataset == 'CIFA10':
            class_data[c_idx].append(d_idx)

    for idx in range(device_num):
        if args.dataset == 'CIFA10':
            data = []
            
            for i in range(cls_per_device):
                tmp = np.random.choice(class_data[(idx + i) % args.num_cls],int(num_data / cls_per_device),-1)
                data = np.concatenate((data, tmp), axis=None)
                
            datas.append(data)
            
    if training:
        self.train_data_idx = np.array(datas)
    else:
        self.test_data_idx = np.array(datas)
