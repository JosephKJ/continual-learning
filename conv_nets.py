import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 200, 10, 4)
        self.bn1 = nn.BatchNorm1d(200)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(200, 400, 3)
        self.bn2 = nn.BatchNorm1d(400)
        self.pool2 = nn.MaxPool1d(4)

        self.avg_pool = nn.AvgPool1d(77)
        self.fc1 = nn.Linear(400, 400)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.avg_pool(x)
        # x = x.permute(0, 2, 1)
        # x = self.fc1(x)

        return x

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        return list
