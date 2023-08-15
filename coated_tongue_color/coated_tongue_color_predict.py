# @Author : guopeng
# coding=utf-8
import json
import os

import torch
import torch.nn as nn
from torchvision.models import resnet152, vgg16
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report


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
                                    nn.Linear(2048, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 5),
                                    )
        # self.trained_model2 = vgg16(pretrained=True)  # .to(device)
        self.model2 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    # 测试一下输出维度[b, 512, 1, 1]
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Dropout(p=0.4),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                                    Flatten(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(12544, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 5)
                                    )
        self.model3 = nn.Sequential(
            nn.Linear(10, 5)
        )

    def forward(self, input1):
        # x1 = self.modelA(input1)
        # x2 = self.modelS(input1)
        # x3 = self.vgg(input1)
        x1 = self.model1(input1)
        x2 = self.model2(input1)
        output1 = torch.cat([x1, x2], dim=1)
        output1 = self.model3(output1)
        return output1


# 加载模型
def load_model():
    model = CoatedTongueColorModel()
    model.load_state_dict(torch.load('best.mdl'))
    return model


def coated_tongue_color_predict(picture_root=r"D:\MyCodes\pythonProject\old_tender\datas\category\0\20.png"):
    """
    预测苔色问题：0代表白色、1代表淡黄色、2代表黄色、3代表焦黄色、4代表灰黑色
    :param picture_root: 图片所在路径
    :return: 类别（格式为整数）
    """
    resize = 224
    preprocessing = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = preprocessing(picture_root)
    img = img.unsqueeze(0)
    use_gpu = torch.cuda.is_available()

    model = load_model()

    if use_gpu:
        device = torch.device('cuda')

        img = img.to(device)
        model = model.to(device)
    with torch.no_grad():
        model.eval()
        logits = model(img)
        label = logits.argmax(dim=1)
    return int(label)


def get_diff_list():
    """
    找到新增图片的名称
    :return: 新增图片名称的列表
    """
    with open(r'D:\MyCodes\pythonProject\datas\data2\st2_total_detail(20220506)_sm_noval_last.json', 'r', encoding='utf-8') as f:
        json_data1 = json.load(f)
        lst1 = list(json_data1.keys())
    with open(r'D:\MyCodes\pythonProject\datas\data2all\second_total_detail(20220808)_sm.json', 'r', encoding='utf-8') as f:
        json_data2 = json.load(f)
        lst2 = list(json_data2.keys())
    lst = []
    img_lst = os.listdir(r'D:\MyCodes\pythonProject\datas\data2all\seg-crop')
    for i in lst2:
        if i not in lst1:
            if (i+'.jpg' in img_lst) or (i+'.png' in img_lst):
                lst.append(i)
    return lst


def get_img_labels(lst, task_label='苔色'):
    """
    获取列表中图片的真实标签
    :param lst: 图片名称列表
    :return: 对应的真实标签列表
    """
    labels_lst = []
    with open(r'D:\MyCodes\pythonProject\datas\data2all\second_total_detail(20220808)_sm.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for i in lst:
        vote_list = list(json_data[i][task_label].values())
        label = vote_list.index(max(vote_list))
        labels_lst.append(label)
    return labels_lst


def predict_new_imgs():
    """
    一簇额
    :return:
    """
    pred_labels = []
    lsts = get_diff_list()
    labels_lst = get_img_labels(lsts)
    print(lsts)
    print(labels_lst)
    for i, lst in enumerate(lsts):
        try:
            img_name = lst + '.png'
            img_dir = os.path.join(r'D:\MyCodes\pythonProject\datas\data2all\seg-crop', img_name)
            pred_label = coated_tongue_color_predict(img_dir)
            pred_labels.append(pred_label)
        except FileNotFoundError:
            img_name = lst + '.jpg'
            img_dir = os.path.join(r'D:\MyCodes\pythonProject\datas\data2all\seg-crop', img_name)
            pred_label = coated_tongue_color_predict(img_dir)
            pred_labels.append(pred_label)
        else:
            pred_labels.append(labels_lst[len(pred_labels)])
        print(lst, "真实值：", labels_lst[i], "预测值：", pred_label)
    print(classification_report(labels_lst, pred_labels))


if __name__ == "__main__":
    # a = coated_tongue_color_predict(r'D:\MyCodes\pythonProject\coated_tongue_color\datas\graycoated\g4.png')
    # print(a)

    dir = r'D:\360MoveData\Users\GP1\Desktop\4'

    for i in os.listdir(dir):
        a = coated_tongue_color_predict(os.path.join(dir, i))
        # if int(a) == 3:
        #     a = '焦黄'
        # else:
        #     a = '灰黑'
        print(a, i)
    # predict_new_imgs()

