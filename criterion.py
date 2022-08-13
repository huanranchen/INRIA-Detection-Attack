import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
import torch.distributed as dist


def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    # rt /= nprocs
    return rt


class InfinityLoader():
    def __init__(self, loader):
        raise NotImplementedError
        # loader cannot be activated using iter method. So this class has some problems
        self.loader = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.loader)
            return data
        except:
            self.loader = iter(self.loader)
            return self.__next__()


def get_loss(x: torch.tensor, model: torch.nn.Module):
    criterion = lambda x: F.mse_loss(x, torch.tensor([0.0]).cuda())
    predictions = model(x)
    # print(predictions)
    if len(x) == 0:
        return x.detach()
    pred = predictions[0]
    scores = pred["scores"]
    mask = scores > 0.5
    scores = scores[mask]
    loss = criterion(scores)
    return loss.item()


class GetPatchLoss():
    def __init__(self, model: torch.nn.Module, loader: DataLoader):
        self.model = model
        self.loader = loader
        self.criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def __call__(self, adv_x, total_step=20):
        result = 0
        pbar = self.loader
        for step, image in enumerate(pbar):
            image = image.to(self.device)
            predictions, _ = self.model(image)
            if len(predictions) == 0:
                continue

            # interpolate the patch into images

            for i, pred in enumerate(predictions):
                scores = pred["scores"]
                mask = scores > 0.5
                boxes = pred["boxes"][mask]
                for now_box_idx in range(boxes.shape[0]):
                    now_box = boxes[now_box_idx]
                    by1, bx1, by2, bx2 = scale_bbox(*tuple(now_box.detach().cpu().numpy().tolist()))
                    if not assert_bbox(bx1, by1, bx2, by2):
                        continue
                    now = F.interpolate(adv_x.unsqueeze(0),
                                        size=get_size_of_bbox(bx1, by1, bx2, by2),
                                        mode='bilinear')
                    try:
                        image[i, :, bx1:bx2, by1:by2] = now
                    except:
                        print(image.shape, now.shape)

            # image[:, :, :patch_size[1], :patch_size[2]] = adv_x

            predictions, grads = self.model(image)
            if len(predictions) == 0:
                continue
            scores = []
            for grad in grads:
                mask = grad > 0.5
                scores.append(grad[mask])
            scores = torch.cat(scores, dim=0)
            loss = self.criterion(scores)

            ########### attention!!!!! 222222222  4444444444 !!!!!  #######
            result += reduce_mean(loss).item()
            if step + 1 >= total_step:
                return result / total_step

        return result / len(self.loader)


class TestAttackAcc():
    def __init__(self, model: torch.nn.Module, loader: DataLoader):
        self.model = model
        self.loader = loader
        self.criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def test_accuracy(self, adv_x, total_step=20):
        result = 0
        pbar = self.loader
        for step, image in enumerate(pbar):
            image = image.to(self.device)
            predictions, _ = self.model(image)
            if len(predictions) == 0:
                continue

            num_bbox_before_attack = 0
            # interpolate the patch into images
            for i, pred in enumerate(predictions):
                scores = pred["scores"]
                mask = scores > 0.5
                boxes = pred["boxes"][mask]
                num_bbox_before_attack += torch.sum(mask).item()
                for now_box_idx in range(boxes.shape[0]):
                    now_box = boxes[now_box_idx]
                    by1, bx1, by2, bx2 = scale_bbox(*tuple(now_box.detach().cpu().numpy().tolist()))
                    if not assert_bbox(bx1, by1, bx2, by2):
                        continue
                    now = F.interpolate(adv_x.unsqueeze(0),
                                        size=get_size_of_bbox(bx1, by1, bx2, by2),
                                        mode='bilinear')
                    try:
                        image[i, :, bx1:bx2, by1:by2] = now
                    except:
                        print(image.shape, now.shape)
            # image[:, :, :patch_size[1], :patch_size[2]] = adv_x

            predictions, _ = self.model(image)

            # from VisualizeDetection import visualizaion
            # from utils import tensor2cv2image
            # visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
            # import time
            # time.sleep(2)
            # assert False
            num_bbox_after_attack = 0
            if len(predictions) == 0:
                pass
            else:
                for pred in predictions:
                    scores = pred["scores"]
                    mask = scores > 0.5
                    num_bbox_after_attack += torch.sum(mask).item()
            this_step_attack_acc = 1 - num_bbox_after_attack / num_bbox_before_attack
            result += this_step_attack_acc
            if step + 1 >= total_step:
                return result / total_step

        return result / (step + 1)

    def __call__(self, *args, **kwargs):
        return self.test_accuracy(*args, **kwargs)
