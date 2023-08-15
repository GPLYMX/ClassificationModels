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
from torchvision.models import resnet152, vgg16, vgg16_bn, vgg19_bn, resnet50, resnext101_32x8d, densenet121, \
    densenet201, resnet18, vgg11, mobilenetv3

from read_datas import ReaderDatas, ReaderClassfiedData
from utils import Evaluate
from utils import Flatten, TransformerModel
from custom_models.custom_resnets import Resnet34
from swin_transformer import SwinTransformer
from swin_transformer_v2 import SwinTransformerV2

batchsz = 16
lr = 1e-4

epochs = 500
# device = torch.device('cuda')
torch.manual_seed(1234)
class_sample_counts = [1851, 270, 85, 12]

use_gpu = torch.cuda.is_available()
if use_gpu:

    print('use_GPU:', True)
    device = torch.device('cuda')
else:
    print('use_GPU:', False)
    device = torch.device('cpu')

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


# 定义Transformer模型的编码器和解码器
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TransformerDecoder, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


# 定义完整的Transformer模型，包括编码器和解码器
class Transformer(nn.Module):
    def __init__(self, img_size, num_layers, hidden_size, num_heads, dropout, num_classes):
        super(Transformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.encoder = TransformerEncoder(num_layers, hidden_size, num_heads, dropout)
        self.decoder = TransformerDecoder(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.mean(dim=0)

        x = self.decoder(x)
        return x


class SelfDefineModel(nn.Module):

    def __init__(self):
        super(SelfDefineModel, self).__init__()
        # self.trained_model = mobilenetv3.MobileNetV3()  # .to(device)
        # self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
        #                             Flatten(),
        #                             nn.Dropout(p=0.4),
        #                             nn.Linear(2048, 128),
        #                             nn.Dropout(p=0.3),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(128, 8),
        #                             )
        # self.trained_model2 = vgg16(pretrained=True)  # .to(device)
        self.model2 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    # 测试一下输出维度[b, 512, 1, 1]
                                    nn.Dropout(p=0.4),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Dropout(p=0.4),
                                    nn.ReLU(inplace=True),
                                    # nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0.4),
                                    # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    # nn.ReLU(inplace=True),
                                    # nn.Dropout(p=0.4),
                                    # nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    # nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(output_size=(64, 64)),
                                    # Flatten(),
                                    # nn.Linear(12544, 512),
                                    #
                                    # nn.ReLU(),
                                    # nn.Linear(512, 64),
                                    # nn.ReLU(),
                                    # nn.Linear(64, 4)
                                    )

        # self.transformer = Transformer(img_size=64*64, num_layers=4, hidden_size=256, num_heads=16, dropout=0.4, num_classes=4)
        self.transformer = SwinTransformerV2(img_size=64, patch_size=4, in_chans=8, num_classes=4,
                                             embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                             window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                             drop_rate=0.4, attn_drop_rate=0.3, drop_path_rate=0.3, )

        self.model3 = nn.Sequential(
            nn.Linear(12, 4)
        )

    def forward(self, input1):
        # x1 = self.model1(input1)

        x2 = self.model2(input1)
        x3 = self.transformer(x2)
        # output1 = torch.cat([x1, x3], dim=1)
        # output1 = self.model3(output1)
        return x3


def main():
    train_loader, val_loader, test_loader = data_load()

    model = SelfDefineModel()
    torch.save(model.state_dict(), 'best.mdl')
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

            weights = np.array([1.0, 1.3, 1.1, 1.5])
            weights = torch.from_numpy(weights)
            weights = weights.to(torch.float32)
            criteon = nn.CrossEntropyLoss(weight=weights.to(device), reduction='mean')
            loss = criteon(logit, y)
            coe = torch.tensor(4e-7)
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
