# encoding: utf-8
"""
Angular Triplet Loss
YE, Hanrong et al, Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification
"""

import torch
from torch import nn
import  torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / torch.norm(x, 2, axis, keepdim=True).expand_as(x).clamp(min=1e-12) # updated by yhr.
    return x


def square_euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12) #.sqrt()  # for numerical stability
    return dist


def hard_example_mining_at(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative pairs, and return the cosine value.
    Args:
      dist_mat: pytorch Variable, NORMALIZED pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      cos_ap: pytorch Variable, cos(anchor, positive); shape [N]
      cos_an: pytorch Variable, cos(anchor, positive); shape [N]

    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    cos_ap = (2 - dist_ap.float()) / 2
    cos_an = (2 - dist_an.float()) / 2

    return cos_ap, cos_an

class expATLoss(object):
    """
    Angular Triplet Loss
    YE, Hanrong et al, Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification
    """

    def __init__(self):
        self.ranking_loss = nn.MarginRankingLoss(margin=1)

    def __call__(self, global_feat, labels):
        global_feat = normalize(global_feat, axis=-1)  # cal dist in cosine distance manner

        # Use hard example mining here for fair comparison with Euclidean bsed triplet loss in this framework.
        square_dist_mat = square_euclidean_dist(global_feat, global_feat) 
        cos_ap, cos_an = hard_example_mining_at(
            square_dist_mat, labels) 

        # cal expAT Loss
        y_true = cos_an.new().resize_as_(cos_an).fill_(1)
        loss = torch.exp(self.ranking_loss(cos_ap, F.relu(cos_an), y_true))

        return loss


