# coding=utf-8
import time
import os
# import ssl

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet152, vgg16, vgg16_bn, vgg19_bn, resnet50, resnext101_32x8d, densenet121, \
    densenet201, resnet18, vgg11, mobilenetv3

from read_datas import ReaderDatas, ReaderClassfiedData
from utils import Evaluate
from Mobile_Former.model import MobileFormer
from utils import Flatten, TransformerModel
from custom_models.custom_resnets import Resnet34
from swin_transformer import SwinTransformer
from swin_transformer_v2 import SwinTransformerV2

batchsz = 16
lr = 1e-4

epochs = 500
class_sample_counts = [1851, 270, 85, 16]

use_gpu = torch.cuda.is_available()
if use_gpu:

    print('use_GPU:', True)
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(3407)
else:
    print('use_GPU:', False)
    device = torch.device('cpu')
    torch.manual_seed(3407)

viz = Visdom()
viz.line([[0., 0., 0.]], [0], win='train', opts=dict(title='loss&f1', legend=['loss', 'f1', 'f11']))


def data_load(file_root=os.path.join('datas', 'data1&data2', 'category'), random_seed=2):
    # train_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="train",
    #                        task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                        label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # val_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="val",
    #                      task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                      label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # test_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="test",cd dat
    #                       task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                       label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    train_db = ReaderClassfiedData(root=file_root, mode='train', random_seed=random_seed, resize=128)
    val_db = ReaderClassfiedData(root=file_root, mode='test', random_seed=random_seed, resize=128)
    test_db = ReaderClassfiedData(root=file_root, mode='test', random_seed=random_seed, resize=128)
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


def Hswish(x, inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.


def Hsigmoid(x, inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.


# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_channels, stride, se='True', nl='HS'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride = stride
        if se:
            self.se = SEModule(exp_channels)
        else:
            self.se = None
        self.conv1 = nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2 = nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=exp_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_channels)
        self.conv3 = nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # 先初始化一个空序列，之后改造其成为残差链接
        self.shortcut = nn.Sequential()
        # 只有步长为1且输入输出通道不相同时才采用跳跃连接(想一下跳跃链接的过程，输入输出通道相同这个跳跃连接就没意义了)
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 下面的操作卷积不改变尺寸，仅匹配通道数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out = self.bn2(self.conv2(out))
            out = self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_large(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16, 3, 16, 1, False, 'RE'),
        (24, 3, 64, 2, False, 'RE'),
        (24, 3, 72, 1, False, 'RE'),
        (40, 5, 72, 2, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (80, 3, 240, 2, False, 'HS'),
        (80, 3, 200, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (112, 3, 480, 1, True, 'HS'),
        (112, 3, 672, 1, True, 'HS'),
        (160, 5, 672, 2, True, 'HS'),
        (160, 5, 960, 1, True, 'HS'),
        (160, 5, 960, 1, True, 'HS')
    ]

    def __init__(self, num_classes=17):
        super(MobileNetV3_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2 = nn.Conv2d(160, 960, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3 = nn.Conv2d(960, 1280, 1, 1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Hswish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = Hswish(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out


class MobileNetV3_small(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16, 3, 16, 2, True, 'RE'),
        (24, 3, 72, 2, False, 'RE'),
        (24, 3, 88, 1, False, 'RE'),
        (40, 5, 96, 2, True, 'HS'),
        (40, 5, 240, 1, True, 'HS'),
        (40, 5, 240, 1, True, 'HS'),
        (48, 5, 120, 1, True, 'HS'),
        (48, 5, 144, 1, True, 'HS'),
        (96, 5, 288, 2, True, 'HS'),
        (96, 5, 576, 1, True, 'HS'),
        (96, 5, 576, 1, True, 'HS')
    ]

    def __init__(self, num_classes=17):
        super(MobileNetV3_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2 = nn.Conv2d(96, 576, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3 = nn.Conv2d(576, 1280, 1, 1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Hswish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.bn2(self.conv2(out))
        se = SEModule(out.size(1))
        out = Hswish(se(out))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out


config_508 = {
    'name': 'mf508',
    'token': 8,  # tokens and embed_dim
    'embed': 192,
    'stem': 24,
    'bneck': {'e': 48, 'o': 24, 's': 1},
    'body': [
        {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 4},
        {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 4},

        {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 4},
        {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 4},

        {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 4},
        {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 4},
        {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 4},
        {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 4},
        {'inp': 240, 'exp': 1440, 'out': 512, 'se': None, 'stride': 1, 'heads': 4},
        {'inp': 512, 'exp': 1440, 'out': 512, 'se': None, 'stride': 1, 'heads': 4},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': 1000  # num_clasess
    ,
}


class SelfDefineModel(nn.Module):

    def __init__(self):
        super(SelfDefineModel, self).__init__()
        self.trained_model = resnet152(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 4),
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
                                    nn.Linear(100, 4)
                                    )
        self.model3 = nn.Sequential(
            nn.Linear(8, 4)
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



def main():
    train_loader, val_loader, test_loader = data_load()

    # model = MobileNetV3_large(num_classes=4)

    model = SelfDefineModel()

    # torch.save(model.state_dict(), 'best.mdl')
    # model.load_state_dict(torch.load('best.mdl'))
    lr = 7e-6
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.to(device)
        criteon = criteon.to(device)

    best_f1 = 0
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        # 生成数据集
        train_loader, val_loader, test_loader = data_load()
        for step, (x, y, img_name) in enumerate(train_loader):
            if use_gpu:
                x, y = x.to(device), y.to(device)

            logit = model(x)

            weights = np.array([1.2, 1.1, 1.1, 1.1])
            weights = torch.from_numpy(weights)
            weights = weights.to(torch.float32)
            criteon = nn.CrossEntropyLoss(weight=weights.to(device), reduction='mean')
            loss = criteon(logit, y)
            coe = torch.tensor(1e-9)
            l2_reg = torch.tensor(0.)
            if use_gpu:
                coe = coe.to(device)
                l2_reg = l2_reg.to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
                loss += coe * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result = Evaluate(model, test_loader)
        f1 = result.return_f1()
        if f1 < 0.35:
            lr = 1e-4
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if f1 > 0.35:
            lr = 7e-5
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if f1 > 0.4:
            lr = 4e-5
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if f1 > 0.6:
            lr = 1e-6
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if f1 > 0.8:
            lr = 5e-7
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if epoch % 1 == 0:
            model.eval()
            result = Evaluate(model, test_loader)
            result.print_error_case()
            result2 = Evaluate(model, train_loader)
            f1 = result.return_acc()

            f11 = result2.return_acc()
            if f1 > best_f1:
                best_epoch = epoch
                best_f1 = f1
                # 保存参数
                torch.save(model.state_dict(), 'best.mdl')
                result.save_classification_report()
            print('epoch:', epoch)
            print('loss：', loss)
            result.print_classification_report()
            result2.print_classification_report()
            # 可视化
            loss_ = loss
            loss_ = loss_.detach().cpu()
            viz.line([[loss_, f1, f11]], [epoch], win='train', update='append')
            time.sleep(0.5)

    # 加载参数
    model.load_state_dict(torch.load('best.mdl'))
    test_acc = Evaluate(model, test_loader).return_acc()
    print(test_acc)
    Evaluate(model, test_loader).print_classification_report()


def test_data2():
    train_db = ReaderDatas(picture_root='.\\datas\\data1&data2\\noval-seg-last', resize=244, mode="train",
                           task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
                           label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    test_db = ReaderDatas(picture_root='.\\datas\\data1&data2\noval-seg-last', resize=244, mode="test",
                          task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
                          label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    new_data = train_db + test_db
    # old_data = ReaderDatas(mode='test', category=['白', '淡黄',  '黄', '焦黄', '灰黑'])
    train_loader = DataLoader(new_data, batch_size=batchsz)
    model = SelfDefineModel()
    model.load_state_dict(torch.load('best.mdl'))
    test_acc = Evaluate(model, train_loader).return_acc()
    print(test_acc)
    Evaluate(model, train_loader).print_classification_report()


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    main()
