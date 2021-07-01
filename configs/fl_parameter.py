class FLParser(object):
    def __init__(self):
        self.model_name = "mobilenet"
        self.dataset = "CIFA10"

        self.pretrain = False
        self.shuffle = True

        self.num_max_devices = 100
        self.num_devices = 10
        self.epsilon = 0.2

        self.num_epochs = 100
        self.cur_epoch = 0
        self.num_local_epochs = 5

        self.lr = 0.003
        self.batch_size = 64
        self.num_download = 10
        self.hyper_var = 1e-3

        self.n = 1000
        self.n_test = 200
        self.seed = 46
        self.num_cls = 10
        self.distribution = 5
