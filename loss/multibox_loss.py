import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import to_var
from utils.bbox_utils import match


class MultiBoxLoss(nn.Module):
    def __init__(self,
                 class_count,
                 threshold,
                 pos_neg_ratio,
                 use_gpu):
        super(MultiBoxLoss, self).__init__()

        self.class_count = class_count
        self.threshold = threshold
        self.pos_neg_ratio = pos_neg_ratio
        self.use_gpu = use_gpu

        self.variance = [0.1, 0.2]

    def log_sum_exp(self, x):
        x_max = x.data.max()
        y = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
        return y

    def forward(self,
                class_preds,
                class_targets,
                loc_preds,
                loc_targets,
                anchors):

        b, num_anchors, _ = loc_preds.shape

        class_m = torch.LongTensor(b, num_anchors)
        loc_m = torch.Tensor(b, num_anchors, 4)

        class_m = to_var(class_m, self.use_gpu)
        loc_m = to_var(loc_m, self.use_gpu)

        for i in range(b):
            class_m[i], loc_m[i] = match(threshold=self.threshold,
                                         class_target=class_targets[i],
                                         loc_target=loc_targets[i],
                                         anchors=anchors.data,
                                         variances=self.variance)

        class_targets = class_m
        loc_targets = loc_m

        pos = class_targets > 0

        # loc loss
        pos_mask = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask].view(-1, 4),
                                    loc_targets[pos_mask].view(-1, 4),
                                    size_average=False)

        # compute max conf across batch for hard negative mining
        batch_conf = class_preds.view(-1, self.class_count)
        class_loss = self.log_sum_exp(batch_conf) - batch_conf.gather(1, class_targets.view(-1, 1))

        # hard negative mining
        class_loss = class_loss.view(b, -1)
        class_loss[pos] = 0
        _, loss_index = class_loss.sort(1, descending=True)
        _, index_rank = loss_index.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.pos_neg_ratio*num_pos, max=pos.shape[1]-1)
        neg = index_rank < num_neg.expand_as(index_rank)

        # class loss including positive and negative examples
        pos_index = pos.unsqueeze(2).expand_as(class_preds)
        neg_index = neg.unsqueeze(2).expand_as(class_preds)
        preds = class_preds[(pos_index+neg_index).gt(0)]
        preds = preds.view(-1, self.class_count)
        targets_weighted = class_targets[(pos+neg).gt(0)]
        class_loss = F.cross_entropy(preds,
                                     targets_weighted,
                                     size_average=False)

        num_matched = num_pos.data.sum()
        class_loss /= num_matched.float()
        loc_loss /= num_matched.float()
        loss = class_loss + loc_loss

        return class_loss, loc_loss, loss
