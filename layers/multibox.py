import torch
import torch.nn as nn


class MultiBox(nn.Module):

    def __init__(self, num_channels, num_anchors, class_count):
        super(MultiBox, self).__init__()

        self.class_modules = nn.ModuleList()
        self.loc_modules = nn.ModuleList()
        self.num_anchors = num_anchors
        self.class_count = class_count

        for _, channel in num_channels:
            self.class_modules.append(self.get_class_subnet(channel))
            self.loc_modules.append(self.get_loc_subnet(channel))

        self.init_weights()

    def get_class_subnet(self, channel):

        layers = []

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=1,
                                stride=1))

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=3,
                                stride=1,
                                padding=1))

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        out_channels = self.class_count * self.num_anchors
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1))

        return nn.Sequential(*layers)

    def get_loc_subnet(self, channel):

        layers = []

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=1,
                                stride=1))

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=channel,
                                kernel_size=3,
                                stride=1,
                                padding=1))

        layers.append(nn.BatchNorm2d(num_features=channel))
        layers.append(nn.ReLU(inplace=True))
        out_channels = 4 * self.num_anchors
        layers.append(nn.Conv2d(in_channels=channel,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1))

        return nn.Sequential(*layers)

    def init_weights(self):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x):

        class_y = []
        loc_y = []

        for i in range(len(x)):
            class_y_temp = self.class_modules[i](x[i].contiguous())
            b = class_y_temp.shape[0]
            class_y_temp = class_y_temp.permute(0, 2, 3, 1).contiguous()
            class_y_temp = class_y_temp.view(b, -1, self.class_count)
            class_y.append(class_y_temp)

            loc_y_temp = self.loc_modules[i](x[i].contiguous())
            b = loc_y_temp.shape[0]
            loc_y_temp = loc_y_temp.permute(0, 2, 3, 1).contiguous()
            loc_y_temp = loc_y_temp.view(b, -1, 4)
            loc_y.append(loc_y_temp)

        class_y = torch.cat(class_y, 1)
        loc_y = torch.cat(loc_y, 1)
        return class_y, loc_y
