from torch import nn


class ClassificationCNN(nn.Module):

    def __init__(self):
        super(ClassificationCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=2),  # 5x30x30
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 5x15x15

            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=2),  # 10x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 10x8x8

            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1),  # 15x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 15x4x4

            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=0),  # 20x2x2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 20x1x1
        )

        self.lin = nn.Sequential(
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, num_flattened_features(x))
        x = self.lin(x)
        return x


def num_flattened_features(x):
    dimension_sizes = x.size()[1:]
    num_features = 1
    for size in dimension_sizes:
        num_features *= size
    return num_features
