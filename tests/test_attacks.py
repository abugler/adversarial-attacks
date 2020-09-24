import numpy as np
import torch
from ae.attacks import FastGraidentSign, SaliencyAttack
from ae.load_mnist import load_mnist_data
from ae.models import ResNet


def softmax_prediction(network, features):
    out = network(features)
    return torch.argmax(out, dim=-1)


def test_fgs():
    _, features, _, labels = load_mnist_data(10, 0, 300)
    network = ResNet()
    network.load_state_dict(torch.load("resnet.pth"))
    loss = torch.nn.NLLLoss
    fgs = FastGraidentSign(network, loss, .1)
    idx = 2
    feature, label = features[idx:idx+1, ...], labels[idx:idx+1, ...]
    p_feature = fgs.perturb(feature, label)
    assert softmax_prediction(network, feature)[0] == 1
    assert softmax_prediction(network, p_feature)[0] == 0


def test_sma():
    _, features, _, labels = load_mnist_data(10, 0, 300)
    network = ResNet()
    network.load_state_dict(torch.load("resnet.pth"))
    sma = SaliencyAttack(network, 10, .1, 10)
    idx = 2

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    feature = features[idx:idx+1, ...].to(device)
    p_feature = sma.perturb(feature, 0, verbose=False)
    assert softmax_prediction(network, feature)[0] == 1
    assert softmax_prediction(network, p_feature)[0] == 0