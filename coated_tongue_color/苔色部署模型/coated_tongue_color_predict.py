# -*- coding: utf-8 -*-
# @Author : guopeng
import time

import torch
import torch.nn as nn
from torchvision.models import resnet152, vgg16
from torchvision import transforms
from PIL import Image


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class CoatedTongueColorModel(nn.Module):

    def __init__(self):
        super(CoatedTongueColorModel, self).__init__()
        self.trained_model = resnet152(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 10),
                                    )
        self.trained_model2 = vgg16(pretrained=True)  # .to(device)
        self.model2 = nn.Sequential(*list(self.trained_model2.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(200, 10)
                                    )
        self.model3 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.model3(x)
        return x


def preprocess_img(original_img):
    """
    :param original_img: RGB（PIL），单张图片
    :return:torch.tensor格式，shape=[b,c,h,w]
    """
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
        self_model = CoatedTongueColorModel()
        self_model.load_state_dict(torch.load('coated_tongue_color_model220829.mdl'))
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device = torch.device('cuda')
            self_model = self_model.to(device)
        return int(self_model(input_img).argmax(dim=1))


def infer(input_img, input_model):
    """
    :param input_img: img格式为RGB（PIL），单张图片
    :param input_model:
    :return:coated tongue color
    """
    index2label = {0: "白", 1: "淡黄", 2: "黄", 3: "焦黄", 4: "灰黑"}
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
        # input_model = input_model.to(device)
        input_img = input_img.to(device)
    index = input_model(input_img)  # 模型可能输出不止一个字段值时，备注各个字段含义
    label = index2label[index]  # 这个必选项

    return label


if __name__ == "__main__":
    # 加载图片为RGB格式
    # model = torch.load("best.mdl")
    t1 = time.time()
    picture_root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data3\category\1\30.png'
    img = Image.open(picture_root).convert('RGB')
    # 返回单个图片
    t2 = time.time()
    img = preprocess_img(img)
    t3 = time.time()
    # 构建模型
    model = Model()
    t4 = time.time()
    # 测试
    pred = infer(img, model)  # pred是具体的中文预测值
    t5 = time.time()
    pred = infer(img, model)
    t6 = time.time()
    print(pred)
    print(t2 - t1)
    print(t3 - t2)
    print(t4 - t3)
    print(t5 - t4)
    print(t6 - t5)
