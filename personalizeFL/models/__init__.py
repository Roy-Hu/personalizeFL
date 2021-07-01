import torch
from torchvision import models
from models.rl_model import PolicyNetwork, QNetwork
from models.mobile import MobileNetV2

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

models = {
    #    "mobilenet": mobilenet_v2(),
    "mobilenet": models.MobileNetV2(),
    "rl": PolicyNetwork,
    "Q": QNetwork,
}
