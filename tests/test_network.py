from ae.models import ResNet
import torch


def test_dont_fail():
    x = torch.rand(1, 1, 28, 28)
    network = ResNet()
    probabilities = network(x)
    assert abs(torch.sum(probabilities).item() - 1) < 1e-4
    assert probabilities.size(-1) == 10
    assert len(probabilities.size()) == 2


