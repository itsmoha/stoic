import torch.nn as nn
from monai.networks.nets import EfficientNetBN


def get_model():
    model = EfficientNetBN(
        "efficientnet-b0",
        spatial_dims=3,
        in_channels=1,
        num_classes=1,
        pretrained=False,
    )
    model._fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=320),
        nn.BatchNorm1d(320),
        nn.SiLU(),
        nn.Linear(320, 1),
    )

    return model
