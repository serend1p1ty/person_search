import torch
import torch.nn as nn
from torch.autograd import Function


class LabeledMatching(Function):

    @staticmethod
    def forward(ctx, feats, pid_labels, lookup_table, momentum=0.5):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we would modify
        # the variable which has the same memory address in backward()
        ctx.save_for_backward(feats, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        score = feats.mm(lookup_table.t())
        return score

    @staticmethod
    def backward(ctx, grad_output):
        feats, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)

        # Update lookup table, but not by standard backpropagation with gradients
        for indx, label in enumerate(pid_labels):
            if label != -1:
                lookup_table[label] = momentum * lookup_table[label] + (1 - momentum) * feats[indx]

        return grad_feats, None, None, None


class LabeledMatchingLayer(nn.Module):
    """Labeled matching for OIM loss function."""

    def __init__(self, num_classes=5532, feat_len=256):
        super(LabeledMatchingLayer, self).__init__()
        self.lookup_table = torch.zeros(num_classes, feat_len).cuda()
        self.num_classes = num_classes
        self.feat_len = feat_len

    def forward(self, feats, pid_labels):
        assert feats.size(1) == self.feat_len, "Feature length does not match."
        labels = pid_labels.clone()
        labels[(labels < 0) | (labels >= self.num_classes)] = -1
        score = LabeledMatching.apply(feats, labels, self.lookup_table)
        return score, labels
