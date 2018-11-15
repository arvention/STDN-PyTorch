import torch
from math import sqrt


anchors_config = {
    '300': [[8, 16, 32, 64, 100, 300],
            [38, 19, 10, 5, 3, 1],
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]]],
    '512': []
}


class AnchorBox(object):

    def __init__(self,
                 new_size):
        super(AnchorBox, self).__init__()

        self.new_size = new_size
        self.steps = anchors_config[str(new_size)][0]
        self.map_sizes = anchors_config[str(new_size)][1]
        self.aspect_ratios = anchors_config[str(new_size)][2]

        self.scales = self.get_scales()

    def get_scales(self, scale_min=0.2, scale_max=1.05):
        scales = [0.1]

        num_map_sizes = len(self.map_sizes)

        scale_diff = scale_max - scale_min

        for k in range(1, num_map_sizes + 1):
            scale = scale_min + scale_diff / (num_map_sizes - 1) * (k - 1)
            scales.append(scale)

        return scales

    def get_boxes(self):
        boxes = []

        for k, map_size in enumerate(self.map_sizes):
            num_elements = map_size ** 2
            size = self.new_size / self.steps[k]
            for i in range(num_elements):

                row = i // map_size
                col = i % map_size

                cx = (col + 0.5) / size
                cy = (row + 0.5) / size

                scale = self.scales[k]
                scale_next = self.scales[k + 1]
                scale_next = sqrt(scale * scale_next)

                boxes += [cx, cy, scale, scale]
                boxes += [cx, cy, scale_next, scale_next]

                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    boxes += [cx, cy, scale * ratio, scale / ratio]
                    boxes += [cx, cy, scale / ratio, scale * ratio]

        output = torch.Tensor(boxes).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output
