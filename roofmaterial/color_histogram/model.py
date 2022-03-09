import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 5, 1)
        # self.conv3 = nn.Conv2d(6, 1, 1)
        self.fc1 = nn.Linear(5, 1)
        # self.fc2 = nn.Linear(3, 1)
        # self.fc1 = nn.Linear(15, 1)
        # self.fc2 = nn.Linear(5, 1)
        # self.fc3 = nn.Linear(5, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)

        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)

        output = torch.sigmoid(x)
        return output.reshape(-1)


if __name__ == '__main__':
    net = Net()
    x = torch.zeros((8, 3, 5, 1))
    print(net(x))