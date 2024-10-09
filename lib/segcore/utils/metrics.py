import torch
from tabulate import tabulate


__all__ = ["Metrics"]


class Metrics:
    def __init__(self, config_manager):
        self.n_classes = config_manager.get("segcore.dataset.n_classes")
        self.class_names = config_manager.get("segcore.dataset.class_names")
        self.table_fmt = config_manager.get("segcore.metrics.table_fmt")
        self.missing_val = config_manager.get("segcore.metrics.missing_val")
        self.ignore_index = config_manager.get("segcore.dataset.ignore_index")
        self.eps = config_manager.get("segcore.metrics.eps")

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.n_classes, self.n_classes, dtype=torch.int64
        ).cuda()
        self.metrics = {}

    def update(self, pred, target):
        pred, target = self.prepare_input(pred, target)
        self.confusion_matrix += torch.bincount(
            self.n_classes * target + pred, minlength=self.n_classes**2
        ).view(self.n_classes, self.n_classes)

    def prepare_input(self, pred, target):
        if pred.dim() > 3:
            pred = torch.argmax(pred, dim=1).view(-1)
        else:
            pred = pred.view(-1)
        target = target.view(-1)
        mask = target != self.ignore_index
        return pred[mask], target[mask]

    def collect(self):
        TP = self.confusion_matrix.diag()
        FP = self.confusion_matrix.sum(0) - TP
        FN = self.confusion_matrix.sum(1) - TP

        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        f1_score = 2 * (precision * recall) / (precision + recall + self.eps)
        iou = TP / (TP + FP + FN + self.eps)

        class_weight = self.confusion_matrix.sum(1) / self.confusion_matrix.sum()

        self.metrics.update(
            {
                "class_weight": class_weight,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

    def create_summary_table(self, metrics):
        class_names = self.class_names if self.class_names else range(self.n_classes)

        tabular_data = []

        for idx, name in enumerate(class_names):
            tabular_data.append(
                [
                    name,
                    *[
                        (
                            f"{metrics[field][idx] * 100:.2f}%"
                            if field != "class_weight"
                            else f"{metrics[field][idx]:.6f}"
                        )
                        for field in metrics
                    ],
                ]
            )

        tabular_data.append(
            [
                "Mean",
                *[
                    (
                        f"{metrics[field].mean() * 100:.2f}%"
                        if field != "class_weight"
                        else None
                    )
                    for field in metrics
                ],
            ]
        )

        headers = ["CLASS NAMES" if self.class_names else "CLASS ID"] + [
            field.upper() for field in metrics
        ]

        return tabulate(
            tabular_data,
            headers,
            tablefmt=self.table_fmt,
            missingval=self.missing_val,
        )

    def __str__(self):
        self.collect()
        return self.create_summary_table(self.metrics)
