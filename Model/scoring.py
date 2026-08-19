# Copyright (c) 2026 Omid Mokhtari

import numpy as np
import torch as pt
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

bc_score_names = ["acc", "ppv", "npv", "tpr", "tnr", "mcc", "auc", "std", "PR", "F1"]


def bc_scoring(
    y_true: pt.Tensor, y_prob: pt.Tensor, threshold: float = 0.5
) -> pt.Tensor:
    y_true_np = y_true.cpu().numpy()  # (N, C)
    y_prob_np = y_prob.cpu().numpy()  # (N, C)
    n_classes = y_true_np.shape[1]

    scores = np.full((10, n_classes), np.nan)

    for i in range(n_classes):
        yt = y_true_np[:, i]
        yp = y_prob_np[:, i]
        yp_bin = (yp >= threshold).astype(int)

        if len(np.unique(yt)) < 2:
            continue

        # ---- DIAGNOSTIC ----
        uniq = np.unique(yt)
        if not set(uniq).issubset({0, 1}):
            print(
                f"[bc_scoring] BAD LABELS class={i} unique_values={uniq} "
                f"counts={np.unique(yt, return_counts=True)}"
            )
        cm = confusion_matrix(yt, yp_bin)
        if cm.shape != (2, 2):
            print(
                f"[bc_scoring] BAD CM SHAPE class={i} shape={cm.shape} "
                f"yt_unique={uniq} yp_bin_unique={np.unique(yp_bin)}"
            )
        # ---------------------

        TN, FP, FN, TP = confusion_matrix(yt, yp_bin).ravel()

        acc = accuracy_score(yt, yp_bin)
        ppv = precision_score(yt, yp_bin, zero_division=np.nan)
        tpr = recall_score(yt, yp_bin, zero_division=np.nan)
        npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
        tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        mcc = matthews_corrcoef(yt, yp_bin)
        auc = roc_auc_score(yt, yp)
        std = np.std(yp)
        pr = average_precision_score(yt, yp)
        f1 = f1_score(yt, yp_bin, zero_division=np.nan)

        scores[:, i] = [acc, ppv, npv, tpr, tnr, mcc, auc, std, pr, f1]

    return pt.from_numpy(scores).float().to(y_true.device)
