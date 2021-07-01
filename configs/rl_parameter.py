class RLParser(object):
    def __init__(self):

        self.model = "rl"
        self.trainer = "td3"
        self.log_dir = "logs"

        self.load = False

        self.max_clamp = 10
        self.val_every = 1
        self.save_every = 1000
        self.decay_every = 50
        self.exploration_epsilon = 0.2

        self.c_lr = 1e-6
        self.a_lr = 1e-6
        self.epochs = 1
        self.discount = 0.0

        self.num_models = 20
        self.num_max_devices = 100
        self.num_devices = 20
        self.topk = 1

        self.FF = 256
        self.state_dim = 440
        self.action_dim = 400
        self.max_action = 1

        self.alpha = 0.5
        self.beta = 0.5
        self.theta = 8

        self.max_ep = 700
        self.init_eps = 0.1
        self.tau = 0.005
        self.policy_noise = 0.2
        self.expl_noise = 0.1

        self.training_step = 1
        self.update_itr = 1
        self.batch_size = 16
        self.replay_buffer_size = 32
        self.explore_steps = 24
        self.episode = 500

        self.epsilon = 0.2
        self.noise_clip = 0.5
        self.imps = 0
