# coding=utf-8
import time
import os
import ssl

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
import numpy as np
from torchvision.models import resnet152, vgg16, vgg16_bn, vgg19_bn, resnet50, resnext101_32x8d, densenet121, \
    densenet201, resnet18, vgg11

from read_datas import ReaderDatas, ReaderClassfiedData, ReaderClassfiedCoatData
from utils import Evaluate
from utils import Flatten
from custom_models.custom_resnets import Resnet34, VGG19_bn, Resnet152, Densenet121, VGG16, Resnet18, VGG13

batchsz = 16
lr = 1e-4

epochs = 500
# device = torch.device('cuda')
torch.manual_seed(123)
class_sample_counts = [1851, 270, 85, 12]

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')

viz = Visdom()
viz.line([[0., 0., 0.]], [0], win='train', opts=dict(title='loss&f1', legend=['loss', 'f1', 'f11']))


def data_load(file_root=os.path.join('datas', 'data1&data2', 'category'), random_seed=2):
    # train_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="train",
    #                        task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                        label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # val_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="val",
    #                      task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                      label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # test_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="test",
    #                       task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                       label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    train_db = ReaderClassfiedData(root=file_root, mode='train', random_seed=random_seed)
    val_db = ReaderClassfiedData(root=file_root, mode='test', random_seed=random_seed)
    test_db = ReaderClassfiedData(root=file_root, mode='test', random_seed=random_seed)
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
        # self.trained_model = vgg11(pretrained=True)  # .to(device)
        # self.vgg = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
        #                          Flatten(),
        #                          nn.Linear(25088, 5000),
        #                          nn.ReLU(),
        #                          nn.Linear(5000, 200),
        #                          nn.ReLU(),
        #                          nn.Linear(200, 4)
        #                          )
        # self.trained_model2 = resnet34(pretrained=True)  # .to(device)
        # self.res34 = nn.Sequential(*list(self.trained_model2.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
        #                            Flatten(),
        #                            # nn.Linear(2048, 512),
        #                            # nn.Dropout(p=0.5),
        #                            # nn.Linear(2048, 512),
        #                            # nn.ReLU(inplace=True),
        #                            nn.Linear(512, 128),
        #                            nn.Dropout(p=0.3),
        #                            nn.ReLU(),
        #                            nn.Linear(128, 4)
        #                            )

        self.trained_model = resnet152(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 4),
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
                                    nn.Linear(12544, 2000),
                                    nn.ReLU(),
                                    nn.Linear(2000, 100),
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

    model = SelfDefineModel()
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
        train_loader, val_loader, test_loader = data_load()
        for step, (x, y, img_name) in enumerate(train_loader):
            if use_gpu:
                x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criteon(logit, y)
            coe = torch.tensor(3e-6)
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
        if f1 > 0.45:
            lr = 1e-6
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if epoch % 1 == 0:
            model.eval()
            result = Evaluate(model, test_loader)
            result.print_error_case()
            result2 = Evaluate(model, train_loader)
            f1 = result.return_f1()

            f11 = result2.return_f1()
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
    train_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="train",
                           task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
                           label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    test_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="test",
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
