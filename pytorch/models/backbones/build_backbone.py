import torch
import torch.nn as nn

def build_backbone(backbone_name, pretrained=True):
    # 간단한 백본 구현
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super(SimpleBackbone, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.relu(self.conv1(x))
            x2 = self.relu(self.conv2(x1))
            x3 = self.relu(self.conv3(x2))
            x4 = self.relu(self.conv4(x3))
            return x1, x2, x3, x4

    return SimpleBackbone() 