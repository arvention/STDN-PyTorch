import torch.nn as nn


class ScaleTransferModule(nn.Module):

    def __init__(self, new_size):
        super(ScaleTransferModule, self).__init__()

        self.new_size = new_size
        self.module_list = self.get_modules()

    def get_modules(self):
        module_list = nn.ModuleList()

        if self.new_size == 300:
            module_list.append(nn.AvgPool2d(kernel_size=9, stride=9))
            module_list.append(nn.AvgPool2d(kernel_size=3, stride=3))
            module_list.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
            module_list.append(nn.PixelShuffle(upscale_factor=2))
            module_list.append(nn.PixelShuffle(upscale_factor=4))

        elif self.new_size == 513:
            module_list.append(nn.AvgPool2d(kernel_size=16, stride=16))
            module_list.append(nn.AvgPool2d(kernel_size=8, stride=8))
            module_list.append(nn.AvgPool2d(kernel_size=4, stride=4))
            module_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
            module_list.append(nn.PixelShuffle(upscale_factor=2))
            module_list.append(nn.PixelShuffle(upscale_factor=4))

        return module_list

    def forward(self, x):

        y = []
        if self.new_size == 300:
            val = 3
        elif self.new_size == 513:
            val = 4

        for i in range(len(x)):
            if i < val:
                y.append(self.module_list[i](x[i]))
            elif i > val:
                y.append(self.module_list[i - 1](x[i]))
            elif i == val:
                y.append(x[i])

        return y
