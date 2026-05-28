import torch
import torch.nn as nn

from resnet3d import ResNet3D


class StructuredFeatureMLP(nn.Module):

    """
    MLP for structured feature fusion.
    """

    def __init__(
        self,
        image_dim=128,
        structured_dim=12,
        num_classes=60
    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                image_dim + structured_dim,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        image_feature,
        structured_feature
    ):

        x = torch.cat(
            [
                image_feature,
                structured_feature
            ],
            dim=1
        )

        return self.layers(x)


class TLINet(nn.Module):

    """
    Multimodal inference model.
    """

    def __init__(
        self,
        num_classes=60,
        structured_dim=12
    ):

        super().__init__()

        self.encoder = ResNet3D()

        self.head = StructuredFeatureMLP(

            image_dim=128,

            structured_dim=structured_dim,

            num_classes=num_classes
        )

    def forward(
        self,
        image,
        structured_feature
    ):

        image_feature = self.encoder(
            image
        )

        pred = self.head(

            image_feature,

            structured_feature
        )

        return pred