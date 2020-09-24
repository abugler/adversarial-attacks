import torch

from ae.train import train
from ae.load_mnist import load_mnist_data
from ae.models import ResNet
from ae.evaluate import find_accuracy


def run():
    resnet = ResNet()
    train_features, test_features, train_targets, test_targets =(
        load_mnist_data(10, fraction=.95, examples_per_class=6000)
    )

    perm = torch.randperm(train_features.size(0))
    val_idx, train_idx = perm[:500], perm[500:]

    # resnet.load_state_dict(torch.load("resnet.pth"))
    resnet = train(resnet, train_features[train_idx], train_targets[train_idx],
                   train_features[val_idx], train_targets[val_idx])
    torch.save(resnet.state_dict(), "resnet.pth")
    print(f"Accuracy: {find_accuracy(resnet, test_features, test_targets)* 100}%")

if __name__ == "__main__":
    run()
