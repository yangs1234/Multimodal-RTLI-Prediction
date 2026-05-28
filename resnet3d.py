import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):

    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        downsample=None
    ):

        super().__init__()

        self.conv1 = conv3x3x3(
            in_planes,
            planes,
            stride
        )

        self.bn1 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(
            planes,
            planes
        )

        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.in_planes = 32

        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                32,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(
            32,
            blocks=2,
            stride=1
        )

        self.layer2 = self._make_layer(
            64,
            blocks=2,
            stride=2
        )

        self.layer3 = self._make_layer(
            128,
            blocks=2,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def _make_layer(
        self,
        planes,
        blocks,
        stride
    ):

        downsample = None

        if stride != 1 or self.in_planes != planes:

            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(planes)
            )

        layers = []

        layers.append(
            BasicBlock(
                self.in_planes,
                planes,
                stride,
                downsample
            )
        )

        self.in_planes = planes

        for _ in range(1, blocks):

            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)

        x = x.flatten(1)

        return x