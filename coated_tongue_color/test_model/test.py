# coding=utf-8
import os
import time
import math

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50, vgg11, resnet152, vgg19, vgg16, resnext101_32x8d, resnet18, \
    densenet201
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

from torchvision import transforms
from PIL import Image
from visdom import Visdom
import numpy as np
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F

from read_datas import ReaderDatas, ReaderClassfiedData
from Mobile_Former.model import MobileFormer
from utils import Evaluate
from utils import Flatten
from custom_models.custom_resnets import Resnet34, VGG19_bn, Resnet152, VGG16
from main import data_load
from main import SelfDefineModel


def main():

    model = SelfDefineModel()
    train_loader, val_loader, test_loader = data_load(file_root=os.path.join('..', 'datas', 'dataAll', 'category'), resize=128)

    print('原始模型：')
    model.load_state_dict(torch.load('best.mdl'))
    result = Evaluate(model, val_loader)
    result.print_classification_report()

    print('剪枝后在测试集')
    model.load_state_dict(torch.load('prun.mdl'))
    result = Evaluate(model, val_loader)
    result.print_classification_report()
    print('剪枝后在训练集')
    result = Evaluate(model, train_loader)
    result.print_classification_report()

    print('半精度模型')
    model = SelfDefineModel().half()
    model.load_state_dict(torch.load('half.mdl'))
    result = Evaluate(model, val_loader, if_CUDA=False)
    result.print_classification_report()


if __name__ == '__main__':
    main()
