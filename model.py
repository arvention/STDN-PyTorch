import torch.nn as nn
from layers.scale_transfer_module import ScaleTransferModule
from layers.multibox import MultiBox
from torchvision.models import densenet169
from layers.detection import Detect


"""
different configurations of STDN
"""
stdn_in = {
    '300': [800, 960, 1120, 1280, 1440, 1664]
}

stdn_out = {
    '300': [(1, 800), (3, 960), (5, 1120), (9, 1280), (18, 360), (36, 104)]
}


class STDN(nn.Module):

    """STDN Architecture"""

    def __init__(self,
                 mode,
                 stdn_config,
                 channels,
                 class_count,
                 anchors,
                 num_anchors,
                 new_size):
        super(STDN, self).__init__()
        self.mode = mode
        self.stdn_in = stdn_in[stdn_config]
        self.stdn_out = stdn_out[stdn_config]
        self.channels = channels
        self.class_count = class_count
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.new_size = new_size

        # self.init_weights()

        self.densenet = densenet169(pretrained=True)
        self.scale_transfer_module = ScaleTransferModule(self.new_size)
        self.multibox = MultiBox(num_channels=self.stdn_out,
                                 num_anchors=self.num_anchors,
                                 class_count=self.class_count)

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(class_count, 200, 0.01, 0.45)

    def get_out_map_sizes(self):
        return [x for x, _ in self.stdn_out]

    def init_weights(self, modules):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        feed forward
        """
        y = self.densenet.features(x)

        output = []
        for stop in self.stdn_in:
            output.append(y[:, :stop, :, :])

        y = self.scale_transfer_module(output)
        class_preds, loc_preds = self.multibox(y)

        if self.mode == 'test':
            output = self.detect(
                self.softmax(class_preds),
                loc_preds,
                self.anchors
            )
        else:
            output = (
                class_preds,
                loc_preds
            )
        return output
