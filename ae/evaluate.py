import torch


def find_accuracy(model, test_features, test_targets):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)

    probabilities = model(test_features)
    pred = torch.argmax(probabilities, -1).reshape(-1)
    accuracy = torch.sum(pred == test_targets).item() / test_targets.size(0)
    return accuracy