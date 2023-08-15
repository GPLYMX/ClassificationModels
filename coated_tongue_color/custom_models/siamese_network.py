# coding=utf-8
import os
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision.models import vgg13_bn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pandas as pd
from MultiRunner import MultiRunner


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.trained_model = resnet18(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(512, 32)
                                    )

    def forward(self, input1, input2):
        output1 = self.model1(input1)
        output2 = self.model1(input2)
        return output1, output2


# 自定义ContrastiveLoss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    同类为0， 不同类为1
    """

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class ReaderSiameseDatas(Dataset):
    """
    一类图片放到一个文件夹里，文件夹名对应图片类别
    """
    def __init__(self, picture_root='..\\datas\\category', mode="train", resize=256, sample_num=2000, ratio=0.7):
        super(ReaderSiameseDatas, self).__init__()
        # 生成的样本数量
        self.sample_num = sample_num
        self.picture_root = picture_root
        # 训练集与验证集之比
        self.ratio = ratio
        # 图片新尺寸大小
        self.resize = resize
        self.mode = mode
        # self.images中存储图片对(路径)，用于训练
        self.images = []
        # self.labels存储self.images的图片对标签，只含0、1
        self.labels = []
        # 用于测试
        self.test_images = []
        self.test_labels = []
        # 训练集，也是用于测试的基准库
        self.base_images = []
        self.base_labels = []
        self.category_num = len(os.listdir(self.picture_root))
        # 多个列表，每个列表中为某一类的图片地址
        self.temp_images = []
        self.temp_labels = []
        self.make_tag()
        self.tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    def make_tag(self):

        for i in range(self.category_num):
            temp_list = []
            for j in os.listdir(os.path.join(self.picture_root, str(i))):
                temp_list.append(str(os.path.join(self.picture_root, str(i), j)))
            temp_list = sorted(temp_list)
            # rand_num = 1
            # random.seed(rand_num)
            # random.shuffle(temp_list)

            self.test_images += temp_list[int(self.ratio*len(temp_list)):]
            self.base_images += temp_list[0:int(self.ratio*len(temp_list))]
            # 根据文件夹的名字打标签(i即为文件夹名)
            for length in range(len(temp_list[int(self.ratio*len(temp_list)):])):
                self.test_labels.append(int(i))
            for length in range(len(temp_list[0:int(self.ratio*len(temp_list))])):
                self.base_labels.append(int(i))
            self.temp_images.append(temp_list[0:int(self.ratio*len(temp_list))])

            # # 检测搭建的网络是否正确
            # self.test_images += temp_list[int(self.ratio * len(temp_list)):]
            # self.base_images += temp_list[0:]
            # for length in range(len(temp_list[int(self.ratio * len(temp_list)):])):
            #     self.test_labels.append(int(i))
            # for length in range(len(temp_list[0:])):
            #     self.base_labels.append(int(i))
            # self.temp_images.append(temp_list[0:])

        # 增加同类图片对和标签
        for i in range(int(self.sample_num)):
            index = random.randint(0, len(self.temp_images)-1)
            if index == 0:
                for j in range(5):
                    self.images.append(random.sample(self.temp_images[index], 2))
                    self.labels.append(int(0))
            else:
                for j in range(5):
                    try:
                        self.images.append(random.sample(self.temp_images[index], 2))
                        self.labels.append(int(0))
                    except ValueError:
                        self.images.append([self.temp_images[index][0], self.temp_images[index][0]])
                        self.labels.append(int(0))
        # 增加非同类图片对和标签
        for i in range(int(self.sample_num)):
            index = random.sample(range(0, len(self.temp_images)), 2)
            self.images.append([random.choice(self.temp_images[index[0]]), random.choice(self.temp_images[index[1]])])
            self.labels.append(int(1))
        # for img in self.temp_images[0]:
        #     for j in range(5):
        #         self.images.append([img, random.choice(self.temp_images[1])])
        #         self.labels.append(int(1))
        #         self.images.append([img, random.choice(self.temp_images[1])])
        #         self.labels.append(int(1))
        #         self.images.append([img, random.choice(self.temp_images[2])])
        #         self.labels.append(int(1))
        # for img in self.temp_images[1]:
        #     for j in range(100):
        #         self.images.append([img, random.choice(self.temp_images[0])])
        #         self.labels.append(int(1))
        #         self.images.append([img, random.choice(self.temp_images[2])])
        #         self.labels.append(int(1))
        # for img in self.temp_images[2]:
        #     for j in range(200):
        #         self.images.append([img, random.choice(self.temp_images[0])])
        #         self.labels.append(int(1))
        #         self.images.append([img, random.choice(self.temp_images[1])])
        #         self.labels.append(int(1))

        # 打散
        rand_num = random.randint(0, 100)
        random.seed(rand_num)
        random.shuffle(self.images)
        random.seed(rand_num)
        random.shuffle(self.labels)

    def get_base_tensors(self):
        """
        读取多维列表中的每一个图片路径， 然后转化为tensor
        :return:
        """
        base_tensors = []
        for lst in self.temp_images:
            temp_tensors = []
            for img in lst:
                img = self.tf(img)
                temp_tensors.append(img)
            base_tensors.append(temp_tensors)
        return base_tensors

    def get_test_tensors(self):
        """
        读取测试列表中的图片路径，转化为tensor
        :return:
        """
        test_tensors = []
        for img in self.test_images:
            img = self.tf(img)
            test_tensors.append(img)
        return test_tensors

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        if self.mode == "train":
            img1, img2, label = self.images[idx][0], self.images[idx][1], self.labels[idx]
            img1 = self.tf(img1)
            img2 = self.tf(img2)
            label = torch.tensor(int(label))
            return img1, img2, label
        if self.mode == "test":
            img, label = self.test_images[idx], self.test_labels[idx]
            img = self.tf(img)
            label = torch.tensor(int(label))
            return img, label


batchsz = 16
lr_ = 4e-5
epochs = 100
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')


def get_vectors(model, tensors):
    """
    模型训练好后，用于生成每个base图片的特征向量
    :param model:
    :param images: 内含多个列表，子列表中存储图片的初始tensor
    :return: 输入每个图片经过模型后的向量
    """
    vectors_list = []
    model.eval()
    # model = model.cup()
    for i in range(len(tensors)):
        temp_vectors = []
        with torch.no_grad():
            for tensor in tensors[i]:
                if torch.cuda.is_available():
                    tensor = tensor.to(device)
                # 模型的输入是四维，因此需要增加一个维度
                tensor = tensor.unsqueeze(0)
                vector, _ = model(tensor, tensor)
                vector = vector.squeeze(0)
                temp_vectors.append(vector)
        vectors_list.append(temp_vectors)
    return vectors_list


def get_predict_label(base_vectors, vector, length=1):
    """

    :param base_vectors: 内含多个子列表，子列表中存储模型输出向量
    :param vector:某张图片经过模型后得到的一维向量
    :return:输出类别（只有一个整数，譬如：0）
    """
    distances = []
    for i in range(len(base_vectors)):
        temp_distances = []
        for vec in base_vectors[i]:
            temp_distances.append(F.pairwise_distance(vec, vector, keepdim=True))
        temp_distances = sorted(temp_distances)
        # print("temp_distances:", temp_distances)
        distances.append(temp_distances)
    label = 0
    dis = sum(distances[0][0:length])/length
    for i in range(1, len(distances)):
        if dis >= distances[i][0]:
            label = int(i)
            dis = distances[i][0]
            continue
    return label


def main():
    train_db = ReaderSiameseDatas()
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False)
    test_db = ReaderSiameseDatas(mode="test")
    base_tensors = test_db.get_base_tensors()
    test_tensors = test_db.get_test_tensors()

    model = SiameseNetwork()
    lr_ = 4e-5
    optimizer = optim.Adam(model.parameters(), lr=lr_)
    criteon = ContrastiveLoss()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
        model = model.to(device)
        criteon = criteon.to(device)

    best_f1 = 0
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        train_db = ReaderSiameseDatas()
        train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False)
        model.train()
        for step, (img1, img2, label) in enumerate(train_loader):
            if use_gpu:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            logit1, logit2 = model(img1, img2)
            loss = criteon(logit1, logit2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:

            # 获取预测标签
            model.eval()
            base_vectors = get_vectors(model, base_tensors)
            predict_label = []
            for img in test_tensors:
                img = img.unsqueeze(0)
                if use_gpu:
                    img = img.to(device)
                a, _ = model(img, img)
                img, _ = model(img, img)
                # if torch.all(torch.eq(img, a)):
                #     print('模型输出结果唯一')
                # else:
                #     print('模型结果不唯一')
                img = img.squeeze(0)
                predict_label.append(get_predict_label(base_vectors, img))
            f1 = f1_score(list(test_db.test_labels), list(predict_label), average='macro')
            if f1 > 0.5:
                lr_ = 1e-5
            else:
                lr_ = 4e-5
            optimizer = optim.Adam(model.parameters(), lr=lr_)
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), 'best.mdl')
                report = classification_report(list(test_db.test_labels), list(predict_label), output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv("repost.csv", index=True)
            print("\n")
            print('损失：', loss)
            print("预测值：", predict_label)
            print("真实值：", test_db.test_labels)
            target_names = ['淡黄', '白', '黄', '灰黑', '焦黄']
            print(classification_report(list(test_db.test_labels), list(predict_label), target_names=target_names))

    # 加载参数
    model.load_state_dict(torch.load('best.mdl'))
    test_f1 = f1_score(list(test_db.test_labels), list(predict_label), average='macro')
    print("test_macro-f1:", test_f1)
    print("最佳模型报告：\n", report)
    print("最佳epoch：", best_epoch)


def read_model_and_prediction():

    use_gpu = torch.cuda.is_available()
    model = SiameseNetwork()
    model.load_state_dict(torch.load('best.mdl'))
    model.eval()
    model = model.cuda()
    train_db = ReaderSiameseDatas()
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False)
    test_db = ReaderSiameseDatas(mode="test")
    base_tensors = test_db.get_base_tensors()
    test_tensors = test_db.get_test_tensors()
    base_vectors = get_vectors(model, base_tensors)
    predict_label = []
    for img in test_tensors:
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        a, _ = model(img, img)
        img, _ = model(img, img)
        # if torch.all(torch.eq(img, a)):
        #     print('模型输出结果唯一')
        # else:
        #     print('模型结果不唯一')
        img = img.squeeze(0)
        predict_label.append(get_predict_label(base_vectors, img))

    print(classification_report(list(test_db.test_labels), list(predict_label)))


if __name__ == "__main__":
    main()

