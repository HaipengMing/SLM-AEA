import math
import torch


def labels_count(labels):
    n = len(labels)
    indices = list(range(n))
    res = {}
    for idx in indices:
        label = labels[idx].item()
        if label in res:
            res[label] += 1
        else:
            res[label] = 1
    return n, indices, res


class CosLr(torch.nn.Module):
    def __init__(self, t_initial, lr_max=0.001, lr_min=1e-6):
        super(CosLr, self).__init__()
        self.t = t_initial
        self.lm = lr_max
        self.ln = lr_min

    def forward(self, x):
        return self.ln + 0.5 * (self.lm - self.ln) * (1 + math.cos(math.pi * x / self.t))
