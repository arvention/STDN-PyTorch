import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import to_var, one_hot_embedding
from utils.bbox_utils import match


class FocalLoss(nn.Module):

    def __init__(self,
                 class_count,
                 threshold,
                 alpha,
                 gamma,
                 use_gpu):
        super(FocalLoss, self).__init__()

        self.class_count = class_count
        self.threshold = threshold
        self.alpha = alpha
        self.gamma = gamma
        self.use_gpu = use_gpu

        self.variance = [0.1, 0.2]

    def focal_loss(self, class_preds, class_targets):

        t = one_hot_embedding(class_targets, self.class_count)
        t = t[:, 1:]  # remove background
        class_preds = class_preds[:, 1:]  # remove background
        p = class_preds.sigmoid()
        pt = torch.where(t > 0, p, 1 - p)    # pt = p if t > 0 else 1-p
        w = (1 - pt).pow(self.gamma)
        w = torch.where(t > 0, self.alpha * w, (1 - self.alpha) * w)
        loss = F.binary_cross_entropy_with_logits(input=class_preds,
                                                  target=t,
                                                  weight=w,
                                                  size_average=False)
        return loss

    def forward(self,
                class_preds,
                class_targets,
                loc_preds,
                loc_targets,
                anchors):
        b, num_anchors, _ = loc_preds.shape

        class_m = torch.Tensor(b, num_anchors)
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

        num_matched = pos.data.long().sum()

        # loc_loss
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask],
                                    loc_targets[pos_mask],
                                    size_average=False)

        # class loss
        class_preds = class_preds.view(-1, self.class_count)
        class_targets = class_targets.view(-1)
        class_loss = self.focal_loss(class_preds, class_targets)

        class_loss /= num_matched.float()
        loc_loss /= num_matched.float()

        loss = class_loss + loc_loss

        return class_loss, loc_loss, loss
