import numpy as np
import torch
import torchvision
from skimage.io import imread
from VisualizeDetection import visualizaion
from models import ssdlite320_mobilenet_v3_large_with_shakedrop
from model.faster_rcnn import fasterrcnn_resnet50_fpn
from Attack import attack_detection, patch_attack_detection, SAM_patch_attack_detection, \
    AttackWithPerturbedNeuralNetwork, patch_attack_classification_in_detection, \
    patch_attack_detection_strong_augment
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.distributed as dist
import os
import argparse
from criterion import TestAttackAcc
from models import faster_rcnn_resnet50_shakedrop
from Draws.DrawUtils.D2Landscape import D2Landscape
from criterion import GetPatchLoss
from tqdm import tqdm
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

device = torch.device("cuda", local_rank)

from data.data import get_loader


def attack():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    # model = faster_rcnn_resnet50_shakedrop()
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    loader = get_loader(train_path="/home/chenhuanran/data/INRIATrain/pos/", batch_size=8)
    #patch_attack_classification_in_detection(model, loader, attack_epoch=10000, attack_step=999999999)
    patch_attack_detection_strong_augment(model, loader, attack_epoch=10000, attack_step=999999999)
    # SAM_patch_attack_detection(model, loader, attack_epoch=3, attack_step=999999999)
    # w = AttackWithPerturbedNeuralNetwork(model, loader)
    # w.test_perturb_strength()


def draw_2d(dataset_path, model, coordinate=None, patch=None):
    loader = get_loader(dataset_path)

    if patch is None:
        patch = torch.load('patch.pth').to(device)
    loss = GetPatchLoss(model, loader)
    d = D2Landscape(loss, patch, mode='2D')
    d.synthesize_coordinates()
    if coordinate is not None:
        d.assign_coordinates(*coordinate)
    d.draw()
    # plt.savefig('landscape.jpg')


def draw_train_test_2d():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    train_path = "/home/chenziyan/work/data/coco/train/train2017/"
    test_path = "/home/chenziyan/work/data/coco/test/test2017/"
    draw_2d(train_path, model)
    draw_2d(test_path, model)
    plt.savefig('landscape.jpg')


def draw_multi_model_2d():
    train_path = "/home/chenziyan/work/data/coco/train/train2017/"
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    draw_2d(train_path, model)

    model = ssdlite320_mobilenet_v3_large(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    draw_2d(train_path, model)
    plt.legend(['faster rcnn', 'ssd'])
    plt.savefig('landscape.jpg')


def draw_by_given_direction():
    train_path = "/home/chenhuanran/data/INRIATest/"
    patch_ensemble = torch.load('patch_ensemble.pth').to(device).detach()
    patch_PGD = torch.load('patch_PGD.pth').to(device).detach()
    coordinate = patch_ensemble - patch_PGD
    coordinate = (coordinate, coordinate)  # this is a bug

    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    draw_2d(train_path, model, patch=patch_PGD, coordinate=coordinate)

    train_path = "/home/chenhuanran/data/INRIATrain/pos/"
    draw_2d(train_path, model, patch=patch_PGD, coordinate=coordinate)
    plt.legend(['test', 'train'])
    plt.savefig('landscape.jpg')


def test_accuracy():
    '''
    estimate on test set
    :return:
    '''
    train_path = "/home/chenziyan/work/data/INRIAPerson/Test/pos/"
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    loader = get_loader(train_path, batch_size=16)
    w = TestAttackAcc(model, loader)
    patch = torch.load('patch.pth').to(device)
    print(w.test_accuracy(patch, total_step=100))


def draw_loss_of_scale(patch):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    loader = get_loader(train_path="/home/chenhuanran/data/INRIATest/", batch_size=8)
    losser = TestAttackAcc(model, loader)

    aspect_ratios = np.arange(0.2, 1.1, 0.1)
    loss = []

    for i in tqdm(range(aspect_ratios.shape[0])):
        aspect_ratio = aspect_ratios[i]
        now_patch = resize_patch_by_aspect_ratio(patch, aspect_ratio)
        loss.append(losser(now_patch))

    loss = np.array(loss)
    plt.plot(aspect_ratios, loss)


def draw_all_picture_of_aspect_ratio(path='./aug/'):
    legends = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                patch = imread(os.path.join(root, file))
                patch = image_array2tensor(patch).cuda().squeeze()
                draw_loss_of_scale(patch)
                legends.append(file[:-4])
    plt.legend(legends)
    plt.savefig('aspect_ratios')


attack()

# image = imread('1.jpg')
# x = image_array2tensor(np.array(image))
# x = attack_detection(x, model, 40)
# pre = model(x)
# image = tensor2cv2image(x)
# visualizaion(pre, image)

# from Draws.DrawUtils.D2Landscape import D2Landscape
# from criterion import get_loss
#
# figure = plt.figure()
# axes = Axes3D(figure)
# wtf = D2Landscape(lambda a: get_loss(a, model), x)
# wtf.synthesize_coordinates()
# wtf.draw(axes=axes)
# plt.show()
# plt.savefig("mypatchlandscape.png")
