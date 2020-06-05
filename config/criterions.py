import torch.nn.functional as F
import torch
from torch import nn
import settings

class expATLoss():
    def __init__(self):
        self.marginloss = torch.nn.MarginRankingLoss(margin = settings.at_margin)

    def forward(self, anc, pos, neg):
        cos_pos = F.cosine_similarity(anc, pos)
        cos_neg = F.relu(F.cosine_similarity(anc, neg))
        y_true = anc.new().resize_as_(anc).fill_(1)[:,0:1]
        return torch.exp(self.marginloss(cos_pos, cos_neg.float(), y_true)) # max(0, -1*(cos_pos - cos_neg))


class CrossEntropyLabelSmoothLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
