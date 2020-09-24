import torch
from ae.train import MNIST


def test_mnist_dataset():
    features = torch.rand(10, 28, 28)
    labels = torch.randint(10, (10,))
    mnist = MNIST(features, labels)
    assert torch.allclose(mnist[0]['features'], features[0, ...])
