# 초기 컨볼루션 레이어 (3x3 커널, 스트라이드 2)
# 스테이지별로 4개의 스테이지가 있으며, 각 스테이지에서 레지듀얼 블록이 여러 번 반복됩니다.
# 각 레지듀얼 블록은 두 개의 컨볼루션 레이어와 Squeeze-and-Excitation(SE) 블록을 포함합니다.
# 각 스테이지의 첫 번째 레지듀얼 블록에는 다운샘플링이 포함됩니다.
# 최종적으로 글로벌 평균 풀링 레이어와 완전 연결 레이어를 적용하여 출력을 얻습니다.

import torch
import torch.nn as nn

class SE(nn.Module): # Squeeze-and-Excitation(SE) 블록
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SE(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class RegNetY_400MF(nn.Module):
    def __init__(self, num_classes=1000):
        super(RegNetY_400MF, self).__init__()

        self.in_channels = 32
        self.stage_channels = [32, 64, 160, 384]
        self.stage_blocks = [1, 2, 6, 6]

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(self.stage_channels[0], self.stage_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.stage_channels[1], self.stage_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.stage_channels[2], self.stage_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.stage_channels[3], self.stage_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage_channels[-1], num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [Bottleneck(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Create an instance of the model
model = RegNetY_400MF(num_classes=1000)
