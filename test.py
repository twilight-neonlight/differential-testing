"""
test.py
-------
DeepXplore를 두 ResNet50 모델에 실행하고 결과를 시각화한다.

실행 전 train.py로 model_a.pth, model_b.pth를 생성해야 한다.
결과 PNG는 results/ 디렉터리에 저장된다.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 저장 가능
import matplotlib.pyplot as plt

from model import get_resnet50
from deepxplore import NeuronCoverageTracker, generate_test_inputs

# ── 설정 ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATHS = ["model_a.pth", "model_b.pth"]
RESULTS_DIR = "results"
NUM_SEEDS = 100   # 시드 입력 수
STEPS = 50        # gradient ascent 반복 횟수
STEP_SIZE = 0.01
LAMBDA = 0.5      # coverage/disagreement 균형

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_models():
    models = []
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} 없음. 먼저 train.py를 실행하세요."
            )
        m = get_resnet50().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
        print(f"Loaded: {path}")
    return models


def get_seed_inputs(n=100):
    """CIFAR-10 테스트셋에서 시드 입력을 가져온다."""
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.tolist()


def save_visualization(results, max_show=10):
    """disagreement 입력을 PNG로 저장한다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = min(len(results), max_show)
    if n == 0:
        print("저장할 disagreement 입력이 없습니다.")
        return

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for i, res in enumerate(results[:n]):
        img = res["input"].squeeze().permute(1, 2, 0).numpy()
        img = img.clip(0, 1)

        preds = res["predictions"]
        pred_labels = [CIFAR10_CLASSES[p] for p in preds]

        axes[i].imshow(img)
        axes[i].set_title(
            "\n".join([f"M{j+1}: {pred_labels[j]}" for j in range(len(preds))]),
            fontsize=8
        )
        axes[i].axis("off")

    plt.suptitle(f"Disagreement-inducing inputs (total: {len(results)})", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "disagreements.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print(f"Device: {DEVICE}")

    # 모델 로드
    models = load_models()

    # 뉴런 커버리지 트래커 (모델 A 기준)
    tracker = NeuronCoverageTracker(models[0], threshold=0.5)

    # 시드 입력 로드
    seeds, _ = get_seed_inputs(NUM_SEEDS)
    print(f"Seed inputs: {len(seeds)}")

    # DeepXplore 실행
    print("Running DeepXplore...")
    results = generate_test_inputs(
        models=models,
        seed_inputs=seeds,
        steps=STEPS,
        step_size=STEP_SIZE,
        lam=LAMBDA,
    )

    # 커버리지 측정을 위해 생성된 입력을 한번 더 순전파
    if results:
        generated = torch.cat([r["input"] for r in results]).to(DEVICE)
        with torch.no_grad():
            models[0](generated)

    coverage = tracker.coverage()
    tracker.remove_hooks()

    # 결과 출력
    print(f"\n{'='*40}")
    print(f"Disagreement-inducing inputs: {len(results)} / {NUM_SEEDS}")
    print(f"Neuron coverage (model A):    {coverage:.2%}")
    print(f"{'='*40}\n")

    # 시각화 저장
    save_visualization(results)

    # 개별 파일로도 저장 (최소 5개)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for idx, res in enumerate(results[:10]):
        img = res["input"].squeeze().permute(1, 2, 0).numpy().clip(0, 1)
        preds = res["predictions"]
        pred_labels = [CIFAR10_CLASSES[p] for p in preds]

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img)
        title = " vs ".join(pred_labels)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        path = os.path.join(RESULTS_DIR, f"disagreement_{idx+1:02d}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Individual PNGs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
