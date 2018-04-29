import torch
import torch.nn as nn


class PairwiseDistance(nn.Module):
    def __init__(self):
        super(PairwiseDistance, self).__init__()

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        dist = (x1 - x2).square().sum(axis=1)
        return dist


class TripletMarginLoss(nn.Module):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance()

    def forward(self, anchor, positive, negative):
        d_p = self.pdist(anchor, positive)
        d_n = self.pdist(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss