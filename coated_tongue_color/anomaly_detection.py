# -*- coding: utf-8 -*-
# @Author : guopeng
import time
import os

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
import numpy as np
from torchvision.models import resnet152, vgg16, vgg16_bn, vgg19_bn, resnet50, resnext101_32x8d, densenet121, densenet201, resnet18
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from read_datas import ReaderDatas, ReaderClassfiedData, ReaderClassfiedCoatData
from utils import Evaluate
from utils import Flatten
from custom_models.custom_resnets import Resnet34, VGG19_bn, Resnet152, Densenet121, VGG16, Resnet18, VGG13


batchsz = 3
lr = 1e-4

epochs = 500
# device = torch.device('cuda')
torch.manual_seed(1234)
class_sample_counts = [1147, 1061, 120]


use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')


def data_load(file_root=os.path.join('datas', 'data_abnormal', 'category')):
    # train_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="train",
    #                        task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                        label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # val_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="val",
    #                      task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                      label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # test_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="test",
    #                       task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                       label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    train_db = ReaderClassfiedData(root=file_root, mode='train')
    val_db = ReaderClassfiedData(root=file_root, mode='test')
    test_db = ReaderClassfiedData(root=file_root, mode='test')
    # 重采样
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    train_targets = train_db.labels
    samples_weights = weights[train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights,
                                                     num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False, sampler=sampler)  # num_workers=2
    val_loader = DataLoader(val_db, batch_size=batchsz)
    test_loader = DataLoader(test_db, batch_size=batchsz)

    return train_loader, val_loader, test_loader


class SelfDefineModel(nn.Module):

    def __init__(self):
        super(SelfDefineModel, self).__init__()
        self.trained_model2 = resnet152(pretrained=True)  # .to(device)
        self.res34 = nn.Sequential(*list(self.trained_model2.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                   Flatten(),
                                   nn.Linear(2048, 512),
                                   nn.Dropout(p=0.5),
                                   # nn.Linear(2048, 512),
                                   # nn.ReLU(inplace=True),
                                   nn.Linear(512, 128),
                                   nn.Dropout(p=0.3),
                                   nn.ReLU(),
                                   nn.Linear(128, 3)
                                   )

    def forward(self, input1):
        x4 = self.res34(input1)
        return x4


def preprocess_img(picture_root):
    """
    :param picture_root: 图片路径
    :return:torch.tensor格式，shape=[b,c,h,w]
    """
    original_img = Image.open(picture_root).convert('RGB')
    resize = 256
    preprocessing = transforms.Compose([
        # lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_img = preprocessing(original_img)
    processed_img = processed_img.unsqueeze(0)
    return processed_img


class Model:
    """
    读取模型，返回加载好参数的模型
    """

    def __call__(self, input_img):
        self_model = SelfDefineModel()
        self_model.load_state_dict(torch.load('best.mdl'))
        self_model.eval()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device = torch.device('cuda')
            self_model = self_model.to(device)
        return self_model(input_img)


def infer(input_img, input_model):
    """
    :param input_img: img格式为RGB（PIL），单张图片
    :param input_model:
    :return:coated tongue color
    """

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
        # input_model = input_model.to(device)
        input_img = input_img.to(device)
    index = input_model(input_img)  # 模型可能输出不止一个字段值时，备注各个字段含义

    return index


def predict_prob(img_root=r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category\3\440.png'):

    img = preprocess_img(img_root)
    model = Model()
    pred = infer(img, model)
    prob = F.softmax(pred, dim=1)
    return pred


if __name__ == "__main__":
    # 加载图片为RGB格式
    # model = torch.load("best.mdl")
    root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category'
    for i in os.listdir(root):
        for img in os.listdir(os.path.join(root, i)):
            print(i, img, predict_prob(os.path.join(root, i, img)))

