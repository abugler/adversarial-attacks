import torch
from torch.utils.data import DataLoader, Dataset


class MNIST(Dataset):
    """
    Args:
        features (torch.Tensor): Array representing features in shape (N, W, H)
        labels (torch.Tensor): Array representing labels in shape (N,)
    """
    def __init__(self, features, labels, device='cpu'):
        if features.size(0) != labels.size(0):
            raise ValueError
        self.features = features.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {
            'features': self.features[i, ...],
            'labels': self.labels[i]
        }


def train(model, train_features, train_labels,
          val_features, val_labels, lr=.001, epochs=100,
          batch_size=100, stop_after=5):
    """
    Trains a network with SGD
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader = DataLoader(
        MNIST(train_features, train_labels, device),
        batch_size=batch_size, shuffle=True)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    loss_fn = torch.nn.NLLLoss()
    min_val_loss, last_desc = float('inf'), 0
    for i in range(epochs):
        print(f"epoch: {i}")
        for batch in train_dataloader:
            model.zero_grad()
            features, labels = batch["features"], batch["labels"]
            out = model(features)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            out = model(val_features)
            loss = loss_fn(out, val_labels)
            val_loss = loss.item()
        print(f"val_loss: {val_loss}")
        if val_loss < min_val_loss:
            min_val_loss, last_desc = val_loss, 0
        else:
            last_desc += 1
        if last_desc == stop_after:
            break
    print("stopped")
    return model

