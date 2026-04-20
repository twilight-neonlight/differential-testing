"""
generator.py
------------
DeepXplore의 핵심: gradient 기반 test input 생성.

논문 Algorithm 1을 PyTorch로 재구현한다.
목적함수 (joint loss):
    L = lambda * L_coverage - (1 - lambda) * L_diff

    L_diff     : 모델 간 disagreement (한 모델의 예측 신뢰도를 최대화하고
                 나머지 모델들의 해당 클래스 확률을 최소화)
    L_coverage : 아직 미활성화된 뉴런의 pre-activation 합산
    lambda     : coverage와 disagreement의 균형 가중치

입력에 대해 gradient ascent를 반복하여 disagreement를 유도한다.
"""

import torch
import torch.nn.functional as F


def generate_test_inputs(
    models,
    seed_inputs,
    steps=50,
    step_size=0.01,
    lam=0.5,
    threshold=0.5,
):
    """
    seed_inputs를 시작점으로 disagreement를 유도하는 입력을 생성한다.

    Args:
        models      : 리스트[nn.Module], 모두 eval 모드여야 함
        seed_inputs : Tensor (N, C, H, W), 시드 입력 배치
        steps       : gradient ascent 반복 횟수
        step_size   : 한 스텝의 perturbation 크기
        lam         : coverage loss 가중치 (논문의 λ)
        threshold   : 뉴런 활성화 임계값

    Returns:
        results: list of dict — disagreement가 발생한 입력과 예측 정보
    """
    device = seed_inputs.device
    results = []

    # 모델별로 미활성화된 뉴런을 추적하기 위한 activation 캡처
    activation_store = [{} for _ in models]

    def make_hook(model_idx, layer_name):
        def hook(module, input, output):
            activation_store[model_idx][layer_name] = output
        return hook

    # ReLU에 hook 등록
    hooks = []
    for m_idx, model in enumerate(models):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                h = module.register_forward_hook(make_hook(m_idx, name))
                hooks.append(h)

    try:
        for i in range(len(seed_inputs)):
            x = seed_inputs[i:i+1].clone().detach().requires_grad_(True)
            x = x.to(device)

            for _ in range(steps):
                # 모든 모델에 대해 forward pass
                outputs = [model(x) for model in models]
                probs = [F.softmax(o, dim=1) for o in outputs]

                # L_diff: model 0이 가장 높게 예측한 클래스에 대해
                # model 0의 신뢰도를 높이고, 나머지 모델들의 신뢰도를 낮춤
                pred_class = probs[0].argmax(dim=1)
                l_diff = probs[0][0, pred_class] - sum(
                    p[0, pred_class] for p in probs[1:]
                )

                # L_coverage: threshold 미만인 뉴런의 activation 합산
                l_cov = torch.tensor(0.0, device=device)
                for store in activation_store:
                    for act in store.values():
                        # threshold 미만인 뉴런만 합산 (활성화 유도)
                        l_cov = l_cov + act[act < threshold].sum()

                loss = (1 - lam) * l_diff + lam * l_cov

                # gradient ascent: loss를 최대화하는 방향으로 x 업데이트
                loss.backward()
                with torch.no_grad():
                    x_grad = x.grad.sign()
                    x = x + step_size * x_grad
                    x = x.clamp(0, 1).detach().requires_grad_(True)

            # 최종 입력에 대한 각 모델의 예측
            with torch.no_grad():
                preds = [model(x).argmax(dim=1).item() for model in models]

            # disagreement 발생 시 기록
            if len(set(preds)) > 1:
                results.append({
                    "input": x.detach().cpu(),
                    "predictions": preds,
                    "seed_index": i,
                })

    finally:
        for h in hooks:
            h.remove()

    return results
