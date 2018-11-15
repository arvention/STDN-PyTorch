import torch.nn as nn


class ScaleTransferModule(nn.Module):

    def __init__(self):
        super(ScaleTransferModule, self).__init__()

        self.module_list = self.get_modules()

    def get_modules(self):
        module_list = nn.ModuleList()

        module_list.append(nn.AvgPool2d(kernel_size=9, stride=9))
        module_list.append(nn.AvgPool2d(kernel_size=3, stride=3))
        module_list.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
        module_list.append(nn.PixelShuffle(upscale_factor=2))
        module_list.append(nn.PixelShuffle(upscale_factor=4))

        return module_list

    def forward(self, x):

        y = []

        for i in range(len(x)):
            if i < 3:
                y.append(self.module_list[i](x[i]))
            elif i > 3:
                y.append(self.module_list[i - 1](x[i]))
            elif i == 3:
                y.append(x[i])

        return y
