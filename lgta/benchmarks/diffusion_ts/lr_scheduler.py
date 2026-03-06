"""
Learning rate scheduler with warmup for Diffusion-TS training.
Ported from https://github.com/Y-debug-sys/Diffusion-TS
"""

from torch import inf
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateauWithWarmup:
    """ReduceLROnPlateau with linear warmup phase.

    During the first ``warmup`` steps the learning rate is linearly increased
    from the initial optimizer LR to ``warmup_lr``.  After that the standard
    plateau-based reduction logic applies.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
        warmup_lr: float | None = None,
        warmup: int = 0,
    ) -> None:
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            self.min_lrs: list[float] = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.warmup_lr = warmup_lr
        self.warmup = warmup
        self.best: float = 0.0
        self.num_bad_epochs: int = 0
        self.mode_worse: float = 0.0
        self.eps = eps
        self.last_epoch = 0
        self.warmup_lrs: list[float] | None = None
        self.warmup_lr_steps: list[float] | None = None

        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _prepare_for_warmup(self) -> None:
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None

        if self.warmup > self.last_epoch and self.warmup_lrs is not None:
            curr_lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [
                max(0, (self.warmup_lrs[i] - curr_lrs[i]) / float(self.warmup))
                for i in range(len(curr_lrs))
            ]
        else:
            self.warmup_lr_steps = None

    def _reset(self) -> None:
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: float) -> None:
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            if self._is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

    def _reduce_lr(self, epoch: int) -> None:
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    def _increase_lr(self, epoch: int) -> None:
        if self.warmup_lr_steps is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr + self.warmup_lr_steps[i], self.min_lrs[i])
            param_group["lr"] = new_lr

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def _is_better(self, a: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return a > best * (self.threshold + 1.0)
        else:
            return a > best + self.threshold

    def _init_is_better(
        self, mode: str, threshold: float, threshold_mode: str
    ) -> None:
        if mode == "min":
            self.mode_worse = inf
        else:
            self.mode_worse = -inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self._prepare_for_warmup()
