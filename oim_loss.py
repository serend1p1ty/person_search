import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class LabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pids, lookup_table, momentum=0.5):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pids)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        scores = features.mm(lookup_table.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pids = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)

        # Update lookup table, but not by standard backpropagation with gradients
        for indx, pid in enumerate(pids):
            if pid >= 0:
                lookup_table[pid] = momentum * lookup_table[pid] + (1 - momentum) * features[indx]

        return grad_feats, None, None, None


class UnlabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pids, queue, tail):
        # The queue/tail can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pids)
        ctx.queue = queue
        ctx.tail = tail

        scores = features.mm(queue.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pids = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        # Update circular queue, but not by standard backpropagation with gradients
        for indx, pid in enumerate(pids):
            if pid == -1:
                queue[tail, :64] = features[indx, :64]
                tail += 1
                if tail >= queue.size(0):
                    tail -= queue.size(0)

        return grad_feats, None, None, None


class OIMLoss(nn.Module):
    def __init__(self, num_persons=5532, queue_size=5000, feat_len=256):
        super(OIMLoss, self).__init__()
        self.register_buffer("lookup_table", torch.zeros(num_persons, feat_len))
        self.register_buffer("queue", torch.zeros(queue_size, feat_len))
        self.register_buffer("tail", torch.tensor(0))

    def forward(self, features, pids):
        """
                    -2,             background
        Person ID = -1,             foreground but unlabeled person
                    natural number, foreground and labeled person
        """
        labeled_matching_scores = LabeledMatching.apply(features, pids, self.lookup_table)
        labeled_matching_scores *= 10
        unlabeled_matching_scores = UnlabeledMatching.apply(features, pids, self.queue, self.tail)
        unlabeled_matching_scores *= 10
        matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
        pids = pids.clone()
        pids[pids == -2] = -1
        return F.cross_entropy(matching_scores, pids, ignore_index=-1)
