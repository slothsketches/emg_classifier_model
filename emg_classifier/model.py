import torch
import torch.nn as nn


class EMGClassifierCNN(nn.Module):
    def __init__(self, num_classes: int, window_size: int):
        super(EMGClassifierCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        flattened_size = 64 * (window_size // 8)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x
