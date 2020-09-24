import torch.nn as nn


# For classification, we will use a simple 8-layer ResNet
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = self.conv_layer()
        self.c2 = self.conv_layer()
        self.c3 = self.conv_layer()
        self.c4 = self.conv_layer()
        self.c5 = self.conv_layer()
        self.c6 = self.conv_layer()
        self.c7 = self.conv_layer()
        self.c8 = self.conv_layer()
        self.max_pool = nn.MaxPool2d((3, 3), stride=1)
        self.fc = nn.Linear(400, 10)
        self.res_activation = nn.ReLU()
        self.final_activation = nn.Softmax(dim=-1)

    def conv_layer(self):
        return nn.Conv2d(1, 1, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x):

        def res_connection(layers, x):
            out = input
            for layer in layers:
                out = layer(x)
            return self.res_activation(out + x)
        out = res_connection([self.c1, self.c2], x)
        out = self.max_pool(out)
        out = res_connection([self.c3, self.c4], out)
        out = self.max_pool(out)
        out = res_connection([self.c5, self.c6], out)
        out = self.max_pool(out)
        out = res_connection([self.c7, self.c8], out)
        out = self.max_pool(out)

        out = out.flatten(start_dim=1, end_dim=-1)
        out = self.final_activation(self.fc(out))
        return out
