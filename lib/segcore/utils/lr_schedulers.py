import math
from typing import List, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["get_scheduler", "adjust_learning_rate"]


class BaseLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        raise NotImplementedError


class PolyLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_iter: int,
        decay_iter: int = 1,
        power: float = 0.9,
        last_epoch: int = -1,
    ):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return self.base_lrs
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
        return [factor * lr for lr in self.base_lrs]


class WarmupLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iter: int = 500,
        warmup_ratio: float = 5e-4,
        warmup: str = "exp",
        last_epoch: int = -1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self) -> float:
        return (
            self.get_warmup_ratio()
            if self.last_epoch < self.warmup_iter
            else self.get_main_ratio()
        )

    def get_main_ratio(self) -> float:
        raise NotImplementedError

    def get_warmup_ratio(self) -> float:
        assert self.warmup in ["linear", "exp"]
        alpha = self.last_epoch / self.warmup_iter
        return (
            self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
            if self.warmup == "linear"
            else self.warmup_ratio ** (1.0 - alpha)
        )


class WarmupPolyLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        power: float,
        max_iter: int,
        warmup_iter: int = 500,
        warmup_ratio: float = 5e-4,
        warmup: str = "exp",
        last_epoch: int = -1,
    ):
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self) -> float:
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        return (1 - alpha) ** self.power


class WarmupExpLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        interval: int = 1,
        warmup_iter: int = 500,
        warmup_ratio: float = 5e-4,
        warmup: str = "exp",
        last_epoch: int = -1,
    ):
        self.gamma = gamma
        self.interval = interval
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self) -> float:
        real_iter = self.last_epoch - self.warmup_iter
        return self.gamma ** (real_iter // self.interval)


class WarmupCosineLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        max_iter: int,
        eta_ratio: float = 0,
        warmup_iter: int = 500,
        warmup_ratio: float = 5e-4,
        warmup: str = "exp",
        last_epoch: int = -1,
    ):
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self) -> float:
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        return (
            self.eta_ratio
            + (1 - self.eta_ratio)
            * (1 + math.cos(math.pi * self.last_epoch / real_max_iter))
            / 2
        )


def get_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    max_iter: int,
    power: int,
    warmup_iter: int,
    warmup_ratio: float,
) -> Union[PolyLR, WarmupPolyLR, WarmupCosineLR]:
    schedulers = {
        "polylr": lambda: PolyLR(optimizer, max_iter),
        "warmuppolylr": lambda: WarmupPolyLR(
            optimizer, power, max_iter, warmup_iter, warmup_ratio, warmup="linear"
        ),
        "warmupcosinelr": lambda: WarmupCosineLR(
            optimizer, max_iter, warmup_iter=warmup_iter, warmup_ratio=warmup_ratio
        ),
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"Unavailable scheduler name: {scheduler_name}. Available schedulers: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name]()


def adjust_learning_rate(
    optimizer: Optimizer,
    base_lr: float,
    max_iters: int,
    cur_iters: int,
    power: float = 0.9,
    nbb_mult: float = 10,
) -> float:
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = lr * nbb_mult
    return lr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 373800
    sched = WarmupPolyLR(
        optim,
        power=0.9,
        max_iter=max_iter,
        warmup_iter=200,
        warmup_ratio=0.1,
        warmup="linear",
        last_epoch=-1,
    )

    lrs = []
    for _ in range(max_iter):
        lrs.append(sched.get_lr()[0])
        optim.step()
        sched.step()

    plt.plot(np.arange(len(lrs)), np.array(lrs))
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

    plt.savefig("lr.png")
