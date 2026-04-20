"""
train.py
--------
CIFAR-10 ResNet50 모델 학습 모듈.

- 동일한 구조의 모델을 서로 다른 설정(seed, augmentation)으로 학습하여
  differential testing에 사용할 2개의 모델을 생성한다.

저장 파일:
  - model_a.pth
  - model_b.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import get_resnet50

# ── 디바이스 설정 ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 데이터 로더 ─────────────────────────────────────────────────────────
def get_dataloaders(batch_size=128, augment=False):
    """
    CIFAR-10 데이터 로더 생성.

    augment=True일 경우 데이터 다양성을 위해
    RandomCrop, HorizontalFlip을 적용한다.
    """

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.ToTensor()

    transform_test = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader


# ── 학습 루프 ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    """
    한 에폭 학습.

    - model.train() : dropout / batchnorm 학습 모드
    - optimizer.zero_grad() : gradient 초기화
    - backward() → step() : 파라미터 업데이트
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ── 평가 ───────────────────────────────────────────────────────────────
def evaluate(model, loader):
    """
    테스트셋 정확도 계산.

    - model.eval() : dropout / batchnorm 평가 모드
    - torch.no_grad() : gradient 계산 비활성화
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


# ── 메인 학습 함수 ──────────────────────────────────────────────────────
def train_model(seed=0, augment=False, save_path="model.pth", epochs=10):
    """
    단일 모델 학습.

    Args:
        seed      : 랜덤 시드 (모델 차이 유도)
        augment   : 데이터 augmentation 여부
        save_path : 가중치 저장 경로
        epochs    : 학습 에폭 수
    """
    print("=" * 50)
    print(f"Training start (seed={seed}, augment={augment})")
    print(f"Device: {DEVICE}")
    print("=" * 50)

    # 시드 고정 → 재현성 확보
    torch.manual_seed(seed)

    trainloader, testloader = get_dataloaders(augment=augment)

    model = get_resnet50().to(DEVICE)

    # Adam: 빠른 수렴 (과제용으로 충분)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, trainloader, optimizer, criterion)
        acc = evaluate(model, testloader)

        print(f"Epoch {epoch:2d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    # 학습 완료 후 저장
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    model.eval()
    return model


# ── 단독 실행 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    두 개의 서로 다른 모델 생성.

    - model_a : 기본 설정
    - model_b : augmentation + 다른 seed
    """

    train_model(seed=0, augment=False, save_path="model_a.pth")
    train_model(seed=42, augment=True, save_path="model_b.pth")