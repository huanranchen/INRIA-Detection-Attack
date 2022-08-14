import math
from torchvision import transforms
import cv2
import torch
import numpy as np
import datetime


def tensor2numpy_image(*args, **kwargs):
    return tensor2cv2image(*args, **kwargs)


def tensor2cv2image(x: torch.tensor) -> np.array:
    x = x.squeeze()
    x = x.permute(1, 2, 0)
    x = x.cpu().numpy()
    x *= 255
    x = np.uint8(x)
    return x


def image_array2tensor(image: np.array) -> torch.tensor:
    image = np.array(image, dtype=float)
    image /= 255
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image


def scale_bbox(bx1, by1, bx2, by2, ratio=2.23):
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    len_x = cx - bx1
    len_y = cy - by1
    len_x /= ratio
    len_y /= ratio
    return int(cx - len_x), int(cy - len_y), int(cx + len_x), int(cy + len_y)


def get_patch_size(patch: torch.tensor) -> (int, int):
    return patch.shape[1], patch.shape[2]


def scale_bbox_keep_patch_ratio(bx1, by1, bx2, by2, patch: torch.tensor, ratio=0.2):
    '''
    保持patch的长宽比
    :param bx1:
    :param by1:
    :param bx2:
    :param by2:
    :param ratio:
    :return:
    '''
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    length, width = get_patch_size(patch)
    target_area = get_patch_area(bx1, by1, bx2, by2) * ratio
    now_area = length * width
    scale_ratio = math.sqrt(target_area / now_area)
    len_x = length / 2 * scale_ratio
    len_y = width / 2 * scale_ratio
    if int(cx - len_x) < bx1:
        len_x = (bx2 - bx1)
        len_y = target_area / len_x
        len_x /= 2
        len_y /= 2
    elif int(cy - len_y) < by1:
        len_y = (by2 - by1) / 2
        len_x = target_area / len_y
        len_x /= 2
        len_y /= 2
    return int(cx - len_x), int(cy - len_y), int(cx + len_x), int(cy + len_y)


def get_patch_area(bx1, by1, bx2, by2):
    return (bx2 - bx1) * (by2 - by1)


def get_size_of_bbox(bx1, by1, bx2, by2):
    return torch.Size([bx2 - bx1, by2 - by1])


def assert_bbox(bx1, by1, bx2, by2):
    if bx2 - bx1 <= 10:
        return False
    if by2 - by1 <= 10:
        return False
    return True


def clamp(x, min=0, max=1):
    return torch.clamp(x, min, max)


def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()
    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')
    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    return date_str + time_str


def resize_patch_by_aspect_ratio(patch, aspect_ratio):
    length, width = get_patch_size(patch)
    length *= math.sqrt(aspect_ratio)
    width /= math.sqrt(aspect_ratio)
    resizer = transforms.Resize((int(length), int(width)))
    return resizer(patch)
