import numpy as np


def compute_recall(gt, indices, k=10, at=10):
    assert k <= at
    assert len(gt) == len(indices)
    recall = [len(np.intersect1d(gt[i, :k], indices[i, :at])) / k
              for i in range(len(gt))]
    return np.mean(recall)


def _apk(gt, predicted, k=10):
    if len(gt) > k:
        gt = gt[:k]
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in gt and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(gt), k)

def compute_map(gt, indices, k=10):
    return np.mean([_apk(a, p, k) for a, p in zip(gt, indices)])
