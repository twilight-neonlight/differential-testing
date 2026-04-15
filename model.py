# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_resnet50(num_classes=10):
    model = resnet50(weights=None)

    # CIFAR-10에 맞게 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_resnet50()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)