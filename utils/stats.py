from typing import DefaultDict, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


class ICBHI_Metrics:
    def __init__(
        self,
        num_classes: int = 4,
        normal_class_label: int = None,
        mixup: bool = False,
        mini_batch: bool = False,
    ) -> None:
        r"""Evaluation metrics for ICBHI challenge.

        Definitions:
            Sensitivity: The ability of the classifier to identify audio with the disease correctly.
            Specificity: The ability of the classifier to identify audio without the disease correctly.
            ICBHI Score: The average of sensitivity and specificity.

        Confusion Matrix:
            Task 1: Anomaly Cycle Classification

                                (True)          \
                                C,  W, B, N      \
                            C   Cc, Wc, Bc, Nc  \
                            W   Cw, Ww, Bw, Nw  \
                (y_pred)    B   Cb, Wb, Bb, Nb  \
                            N   Cn, Wn, Bn, Nn  \
                (Total-True)    Ct, Wt, Bt, Nt

            Task 2: Respiratory Disease Detection

                                    (True)          \
                                    C,  NC, H        \
                (y_pred)        C   Cc, NCc, Hc     \
                                NC  Cnc, NCnc, Hnc  \
                                H   Ch, NCh, Hh     \
                (Total-True)        Ct, NCt, Ht

        Label:
            TASK1_1 = {"Normal": 0, "Wheezes": 1, "Crackles": 2, "Both": 3} \
            TASK1_2 = {"Normal": 0, "Abnormal": 1} \
            TASK2_1 = {"Healthy": 0, "Chronic": 1, "NonChronic": 2} \
            TASK2_2 = {"Healthy": 0, "Unhealthy": 1}

        Equations:
            Sensitivity Task1-1: (Cc + Ww + Bb) / (Ct + Wt + Bt) \
            Sensitivity Task1-2: (Ccwb + Wcwb + Bcwb) / (Ct + Wt + Bt) \
            Specificity Task1: Nn / Nt 

            Sensitivity Task2-1: (Cc + NCnc) / (Ct + NCt) \
            Sensitivity Task2-2: (Ccnc + NCcnc) / (Ct + NCt) \
            Specificity Task2: Hh / Ht 

            Score: (Sensitivity + Specificity) / 2 \
            Accuracy: (TP + TN) / (TP + TN + FP + FN) 
        """
        self.num_classes = num_classes
        self.normal_class_label = normal_class_label
        self.mixup = mixup
        self.mini_batch = mini_batch

        self._init_attr()

    def update_lists(
        self,
        logits: torch.Tensor = None,
        y_true: torch.Tensor = None,
        y_pred: torch.Tensor = None,
    ) -> None:
        if self.mini_batch:
            self.y_true = torch.cat((self.y_true, y_true.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred, y_pred.detach().cpu()), dim=0)
        else:
            _, y_pred = logits.max(1)
            self.y_true = torch.cat((self.y_true, y_true.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred, y_pred.detach().cpu()), dim=0)
            self.y_pred_prob = torch.cat(
                (self.y_pred_prob, logits.softmax(-1).detach().cpu()), dim=0
            )

    def update_mixup_stats(
        self,
        logits: torch.Tensor,
        y_true_a: torch.Tensor,
        y_true_b: torch.Tensor,
        lam: torch.Tensor,
    ) -> None:
        _, y_pred = logits.max(1)
        self.total += y_pred.size(0)
        correct_hit = (
            lam * y_pred.eq(y_true_a.data).detach().cpu().sum().float()
            + (1 - lam) * y_pred.eq(y_true_b.data).detach().cpu().sum().float()
        )
        self.correct += correct_hit.mean()

    def get_stats(self) -> Tuple[float, float, float, float]:
        r"""Compute the sensitivity, specificity, score and balanced accuracy based on ICBHI challenge definition through confusion matrix.

        Returns:
            acc: The accuracy.
            se: The sensitivity.
            sp: The specificity.
            sc: The score.
        """
        self._compute_confusion_matrix()
        self._compute_icbhi_scores()
        self.acc = self.confusion_matrix.diag().sum() / self.confusion_matrix.sum()
        return self.acc, self.se, self.sp, self.sc

    def get_mixup_stats(self) -> float:
        self.acc = self.correct / self.total
        return self.acc

    def get_precision_recall_fbeta(
        self, average: str = "macro", fbeta: float = 1.0
    ) -> Tuple[
        Union[float, np.array], Union[float, np.array], Union[float, np.array], np.array
    ]:
        r"""Compute the precision, recall and fbeta score.

        Returns:
            `shape = (num_classes,)`
            precision: The precision score.
            recall: The recall score.
            f-beta: The f-beta score.
            support: The number of samples in each class.
        """
        assert average in [
            "macro",
            "micro",
            "weighted",
        ], "average must be macro, micro or weighted"
        return precision_recall_fscore_support(
            y_true=self.y_true,
            y_pred=self.y_pred,
            average=average,
            beta=fbeta,
            zero_division=0,
        )

    def reset_metrics(self) -> None:
        self._init_attr()

    def _init_attr(self) -> None:
        if self.mixup:
            self.correct = 0.0
            self.total = 0
        else:
            self.y_true = torch.tensor([], dtype=torch.long)
            self.y_pred = torch.tensor([], dtype=torch.long)
            self.y_pred_prob = torch.tensor([], dtype=torch.float)
        self.acc = 0.0

    def _compute_confusion_matrix(self):
        self.confusion_matrix = self.num_classes * self.y_true + self.y_pred
        self.confusion_matrix = torch.bincount(self.confusion_matrix)
        if len(self.confusion_matrix) < self.num_classes * self.num_classes:
            self.confusion_matrix = torch.cat(
                (
                    self.confusion_matrix,
                    torch.zeros(
                        self.num_classes * self.num_classes
                        - len(self.confusion_matrix),
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
        self.confusion_matrix = self.confusion_matrix.reshape(
            self.num_classes, self.num_classes
        ).T

    def _compute_icbhi_scores(self):
        self._compute_sp()
        self._compute_se()
        self._compute_sc()

    def _compute_sp(self) -> None:
        self.sp = (
            self.confusion_matrix[self.normal_class_label, self.normal_class_label]
            / self.confusion_matrix[:, self.normal_class_label].sum()
        )

    def _compute_se(self) -> None:
        self.se = (
            self.confusion_matrix.diag().sum()
            - self.confusion_matrix[self.normal_class_label, self.normal_class_label]
        ) / self.confusion_matrix[:, self.normal_class_label + 1 :].sum()

    def _compute_sc(self) -> None:
        self.sc = (self.sp + self.se) * 0.5

def print_stats(stats: Union[Tuple, DefaultDict], names : Tuple = None):
    if isinstance(stats, DefaultDict):
        stat_str = ' '.join([f'{k} : {v}' for k,v in stats])
    elif names is None:
        names = (f'Metric {i}' for i in range(len(stats)))
        stat_str = ' '.join([f'{k} : {v}' for k,v in zip(names,stats)])
    else:
        assert len(stats) == len(names)
        stat_str = ' '.join([f'{k} : {v}' for k,v in zip(names,stats)])
    return stat_str

if __name__ == "__main__":
    cm = ICBHI_Metrics(num_classes=4, normal_class_label=0)
    y_pred = torch.tensor(
        [
            [0.0540, 0.0671, 0.2014, -0.1081],  # 2
            [0.0487, 0.0771, 0.0015, -0.1226],  # 1
            [0.0500, 0.0776, 0.0108, -0.1131],  # 1
            [0.0531, 0.0627, 0.0017, -0.1051],  # 1
            [0.0522, 0.0733, 0.0051, -0.1096],  # 1
            [0.0454, 0.0782, -0.0077, 0.1375],  # 3
            [0.0513, 0.0673, -0.0002, -0.1134],  # 1
            [0.0325, 0.0547, 0.0047, -0.0967],  # 1
            [0.0475, 0.0639, 0.0017, -0.1037],  # 1
            [0.1489, 0.0728, 0.0022, -0.1135],  # 0
        ]
    )
    y_true = torch.tensor([3, 0, 1, 1, 0, 1, 3, 0, 1, 0])
    cm.update_lists(logits=y_pred, y_true=y_true)
    cm.get_stats()
    assert cm.confusion_matrix.shape == (4, 4)
    assert round(cm.se.item(), 4) == 0.5000, "3/6 = 0.5000"
    assert round(cm.sp.item(), 4) == 0.2500, "1/4 = 0.2500"
    assert round(cm.sc.item(), 4) == 0.3750, "se + sp / 2 = 0.3750"
    assert round(cm.acc.item(), 4) == 0.4000, "4/10 = 0.4000"
    print(f"Confusion Matrix: \n{cm.confusion_matrix.numpy()}")
    print(f"==================== Test Passed ====================")
