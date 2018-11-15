import os
import torch
from PIL import ImageDraw
from torchvision import transforms


def to_var(x, use_gpu, requires_grad=False):
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def one_hot_embedding(labels, class_count):
    '''Embedding labels to one-hot.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      class_count: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(class_count, device=labels.device)  # [D,D]
    return y[labels.long()]  # [N,D]


def show(image, labels, new_size):

    transform = transforms.Compose([transforms.ToPILImage()])
    image = image.cpu()
    labels = labels.cpu()
    image = transform(image)

    draw = ImageDraw.Draw(image)
    for label in labels:
        label = [i.item() for i in label]
        label[0] *= new_size
        label[1] *= new_size
        label[2] *= new_size
        label[3] *= new_size
        draw.rectangle(label[0:4], outline=(255, 0, 0))
        draw.text(label[0:2], str(label[4]), fill=(0, 255, 0))

    image.show()
