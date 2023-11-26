from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3, tuning=False):
        super(SRCNN, self).__init__()

        n1, n2, n3 = (64, 32, num_channels);
        k1, k2, k3 = (9, 5, 5)

        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=k2, padding=k2 // 2)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=k3, padding=k3 // 2)
        self.relu = nn.ReLU(inplace=True)

        if not tuning:
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.conv3.weight)

            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.bias)
            nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
