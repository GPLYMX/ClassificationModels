# coding=utf-8
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
from torchvision.models import resnet152, vgg16, vgg16_bn, vgg19_bn, resnet50, resnext101_32x8d, densenet121, \
    densenet201, resnet18
import matplotlib.pyplot as plt

from read_datas import ReaderDatas, ReaderClassfiedData, ReaderClassfiedCoatData
from utils import Evaluate
from utils import Flatten
from custom_models.custom_resnets import Resnet34, VGG19_bn, Resnet152, Densenet121, VGG16, Resnet18, VGG13

batchsz = 16
lr = 5e-4

epochs = 500
# device = torch.device('cuda')
torch.manual_seed(1234)
class_sample_counts = [1147, 1061, 120]

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

viz = Visdom()
viz.line([[0., 0.]], [0], win='train', opts=dict(title='G&D', legend=['G', 'D']))


def data_load(file_root=os.path.join('..', 'datas', 'data_abnormal', 'category')):
    # train_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="train",
    #                        task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                        label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # val_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="val",
    #                      task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                      label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    # test_db = ReaderDatas(picture_root='.\\datas\\data2\\noval-seg-last', resize=244, mode="test",
    #                       task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'],
    #                       label_filename='.\\datas\\data2\\st2_total_detail(20220506)_sm_noval_last.json')
    train_db = ReaderClassfiedData(root=file_root, mode='train', resize=100)
    val_db = ReaderClassfiedData(root=file_root, mode='test', resize=100)
    test_db = ReaderClassfiedData(root=file_root, mode='test', resize=100)
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


# class G(nn.Module):
#     def __init__(self):
#         super(G, self).__init__()
#         self.main = nn.Sequential(nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(3, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                                   nn.ReLU(inplace=True),
#                                   nn.AdaptiveAvgPool2d(output_size=(20, 20)),
#                                   nn.Conv2d(9, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                                   nn.ReLU(inplace=True),
#                                   nn.AdaptiveAvgPool2d(output_size=(100, 100)),
#                                   nn.Conv2d(9, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(9, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                                   nn.Tanh()
#                                   )
#
#     def forward(self, x):
#         img = self.main(x)
#         # img = img.view(-1, 20, 20, 9)
#         return img

# Generator Code

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        nz = 100
        ngf = 100
        nc = 3
        self.ngpu = 1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        nz = 100
        ngf = 100
        ndf = 100
        nc = 3
        self.ngpu = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 初始化模型、优化器、损失函数
gen = G().to(device)
gen.apply(weights_init)
dis = D().to(device)
dis.apply(weights_init)

d_optim = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
g_optim = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

loss_fn = torch.nn.BCELoss()


# 绘图代码
def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    prediction = prediction.transpose(0, 2, 3, 1)
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.savefig(os.path.join('pictures', str(epoch)))
    plt.show(block=False)
    plt.pause(1)  # 显示1s
    plt.close()


test_input = torch.randn(16, 100, 1, 1, device=device)

train_loader, val_loader, test_loader = data_load()
D_loss = []
G_loss = []
for epoch in range(500):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(train_loader)
    for step, (img, lab, name) in enumerate(train_loader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(batchsz, 100, 1, 1, device=device)

        d_optim.zero_grad()
        real_output = dis(img)
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))  # 得到判别器在真实图像上的损失
        d_real_loss.backward()

        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())  # 判别器生成输入图片
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))  # 得到判别器在生成图像上的损失
        d_fake_loss.backward()
        # 两部分加起来就是判别器的损失
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))  # 期望生成的图片全判断为1
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:', epoch)
        gen_img_plot(gen, epoch, test_input)
    viz.line([[g_epoch_loss.cpu(), d_epoch_loss.cpu()]], [epoch], win='train', update='append')
    print(d_epoch_loss, g_epoch_loss)
    torch.save(gen.state_dict(), 'gen.mdl')
    torch.save(dis.state_dict(), 'dis.mdl')
# if __name__ == '__main__':
#     train_loader, val_loader, test_loader = data_load()
#     img, _, _ = next(iter(train_loader))
#     print(img.shape)
#     a = torch.rand(2, 3, 10, 10)
#     print(a.shape)
#     G = G()
#     a = G(a)
#     print(a.shape)
