import torch
import torch.nn as nn
from torch.autograd import Function


class CircularQueue:
    """A simple circular queue with only tail pointer."""

    def __init__(self, queue_size=5000, feat_len=256):
        self.data = torch.zeros(queue_size, feat_len).cuda()
        self.tail = 0
        self.queue_size = queue_size
        self.feat_len = feat_len

    def enqueue(self, item):
        assert item.size(0) == self.feat_len, "Feature length does not match."
        self.data[self.tail] = item
        self.tail = (self.tail + 1) % self.queue_size


class UnlabeledMatching(Function):

    @staticmethod
    def forward(ctx, feats, pid_labels, queue):
        # The queue can't be saved with ctx.save_for_backward(), as we would modify
        # the variable which has the same memory address in backward()
        ctx.save_for_backward(feats, pid_labels)
        ctx.queue = queue

        score = feats.mm(queue.data.t())
        return score

    @staticmethod
    def backward(ctx, grad_output):
        feats, pid_labels = ctx.saved_tensors
        queue = ctx.queue

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        # Update circular queue, but not by standard backpropagation with gradients
        for indx, label in enumerate(pid_labels):
            if label == -1:
                queue.enqueue(feats[indx])

        return grad_feats, None, None


class UnlabeledMatchingLayer(nn.Module):
    """Unlabeled matching for OIM loss function."""

    def __init__(self):
        super(UnlabeledMatchingLayer, self).__init__()
        self.queue = CircularQueue()

    def forward(self, feats, pid_labels):
        score = UnlabeledMatching.apply(feats, pid_labels, self.queue)
        return score
