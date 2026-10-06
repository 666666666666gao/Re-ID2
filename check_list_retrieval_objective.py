"""CPU value/gradient/permutation checks against a scalar Smooth-AP definition."""
import json

import torch
from torch.nn import functional as F

from list_retrieval_objective import smooth_ap


def reference(features, labels):
    unit = F.normalize(features.float(), dim=1)
    scores = unit @ unit.T
    aps = []
    for i in range(len(labels)):
        positives = [k for k in range(len(labels)) if k != i and labels[k] == labels[i]]
        precisions = []
        for k in positives:
            all_rank = 1 + sum(torch.sigmoid((scores[i, j] - scores[i, k]) / .01)
                               for j in range(len(labels)) if j != i and j != k)
            positive_rank = 1 + sum(torch.sigmoid((scores[i, j] - scores[i, k]) / .01)
                                    for j in positives if j != k)
            precisions.append(positive_rank / all_rank)
        aps.append(torch.stack(precisions).mean())
    return 1 - torch.stack(aps).mean()


def main():
    torch.manual_seed(42)
    features = torch.randn(12, 32, requires_grad=True)
    labels = torch.tensor([2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1])
    actual = smooth_ap(features, labels)
    expected = reference(features, labels)
    gradient = torch.autograd.grad(actual, features, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, features)[0]
    value_error = float((actual - expected).detach().abs())
    gradient_error = float((gradient - expected_gradient).abs().max())
    assert value_error < 1e-6 and gradient_error < 1e-5
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    permutation = torch.randperm(len(labels))
    permutation_error = float((actual.detach() - smooth_ap(features.detach()[permutation], labels[permutation])).abs())
    assert permutation_error < 1e-6
    separated = torch.eye(3).repeat_interleave(4, 0)
    tied = torch.ones(12, 3)
    ordered_labels = torch.arange(3).repeat_interleave(4)
    perfect_loss = float(smooth_ap(separated, ordered_labels))
    tie_loss = float(smooth_ap(tied, ordered_labels))
    assert perfect_loss < 1e-6 and tie_loss > perfect_loss
    print(json.dumps(dict(status='PASS_CPU_LIST_VALUE_GRADIENT_AND_PERMUTATION',
        value_error=value_error, gradient_error=gradient_error, permutation_error=permutation_error,
        perfect_loss=perfect_loss, tie_loss=tie_loss, gradient_l1=float(gradient.abs().sum()),
        optimizer_updates=0, neural_encoder_calls=0, device='cpu')))


if __name__ == '__main__':
    main()
