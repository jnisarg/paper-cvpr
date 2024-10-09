import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Criterion"]


class Criterion(nn.Module):
    def __init__(self, config_manager):
        super(Criterion, self).__init__()

        self.n_classes = config_manager.get("segcore.dataset.n_classes")

        self.ohem_ratio = config_manager.get("segcore.criterion.ohem_ratio")
        self.use_ohem = self.ohem_ratio > 0 and self.ohem_ratio < 1
        self.ignore_index = config_manager.get("segcore.dataset.ignore_index")

        self.weights = torch.tensor(
            config_manager.get("segcore.criterion.weights", [1.0] * self.n_classes)
        )

    def forward(self, pred, target):
        loss = F.cross_entropy(
            pred,
            target,
            weight=self.weights.to(pred.dtype).to(pred.device),
            reduction="none",
            ignore_index=self.ignore_index,
        )

        loss = loss.view(-1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        loss = loss[valid_mask]

        if self.use_ohem:
            sorted_loss, _ = torch.sort(loss, descending=True)

            n_hard_examples = int(self.ohem_ratio * sorted_loss.size(0))
            hard_loss = sorted_loss[:n_hard_examples]

            return hard_loss.mean()

        return loss.mean()
