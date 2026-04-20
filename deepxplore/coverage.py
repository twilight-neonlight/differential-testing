"""
coverage.py
-----------
뉴런 커버리지(Neuron Coverage) 측정 모듈.

DeepXplore 논문 Section 3.1:
    threshold t를 넘는 출력을 낸 뉴런을 "activated"로 간주하고,
    전체 뉴런 중 한 번이라도 activated된 비율을 커버리지로 정의한다.
"""

import torch


class NeuronCoverageTracker:
    def __init__(self, model, threshold=0.5):
        """
        Args:
            model     : 커버리지를 추적할 PyTorch 모델
            threshold : 뉴런 활성화 판정 임계값
        """
        self.threshold = threshold
        self.covered = {}   # layer_name -> set of activated neuron indices
        self.total = {}     # layer_name -> total neuron count
        self._hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        """ReLU 뒤 activation을 캡처하기 위해 forward hook을 등록한다."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            # output shape: (batch, channels, H, W) or (batch, features)
            # 배치 평균 activation으로 판정
            act = output.detach()
            if act.dim() > 2:
                # conv layer: 공간 평균 후 채널별 판정
                act = act.mean(dim=(2, 3))  # (batch, channels)

            # 배치 내 어느 샘플이든 threshold를 넘으면 activated로 기록
            activated = (act > self.threshold).any(dim=0)  # (neurons,)

            # 같은 ReLU 인스턴스가 다른 shape으로 재사용될 때(e.g. Bottleneck)
            # 뉴런 수를 키에 포함해 별도로 추적한다
            key = f"{name}_{activated.numel()}"
            if key not in self.covered:
                self.covered[key] = set()
                self.total[key] = activated.numel()

            indices = activated.nonzero(as_tuple=True)[0].tolist()
            self.covered[key].update(indices)

        return hook

    def coverage(self):
        """현재까지의 전체 뉴런 커버리지(0~1)를 반환한다."""
        total_covered = sum(len(v) for v in self.covered.values())
        total_neurons = sum(self.total.values())
        if total_neurons == 0:
            return 0.0
        return total_covered / total_neurons

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
