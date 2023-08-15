# coding=utf-8
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


class Evaluate:

    def __init__(self, model, loader, TopN=1, threshold=0.2, task_label="苔色"):
        self.pred = []
        self.y = []
        self.label_list = []
        self.img_names = []
        self.TopN = TopN
        self.threshold = threshold
        self.task_label = task_label
        correct = 0
        total = len(loader.dataset)
        model.eval()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
        for x, y, img_name in loader:
            # x, y = x.to(device), y.to(device)
            if use_gpu:
                device = torch.device('cuda')
                x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1)
                self.y += y.tolist()
                if self.TopN <= 1:
                    self.pred += pred.tolist()
                else:
                    # img_name的格式是('20220522002933-1.png',)
                    self.pred += self.get_topn_label(pred.tolist(), img_name)
                self.img_names += img_name
            correct += torch.eq(pred, y).sum().float().item()

            if y.tolist() not in self.label_list:
                self.label_list.append(y.tolist())

        self.acc = correct / total

    def get_topn_label(self, pred_label, img_names):
        # pred_label是一个含多个个元素的列表
        mark_label = pred_label
        with open(os.path.join('datas', 'data2', 'st2_total_detail(20220506)_sm_noval_last.json'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            category_num = len(json_data[list(json_data.keys())[0]][self.task_label])
            for idx, img_name in enumerate(img_names):
                if img_name[-4:] == '.png':
                    img_name = img_name[:-4]
                try:
                    label, confidence_index, confidence = self.extract_label(json_data[img_name][self.task_label], category_num)
                    for i in range(len(label)):
                        if (pred_label[idx] == label[i]) & (confidence_index[i] >= self.threshold):
                            # if i != 0:
                                # print(True)
                            mark_label[idx] = label[0]
                            break
                except Exception as e:
                    pass
        return mark_label

    def extract_label(self, string, category_num):
        """
        label表示前topN的一个列表
        """
        confidence = True
        confidence_index = []
        vote_num = list(string.values())
        label = list(pd.Series(vote_num).sort_values(ascending=False).index[:self.TopN])
        for i in range(len(label)):
            confidence_index.append(vote_num[label[i]] / sum(vote_num))

        if (confidence_index[0] <= 0.5) & (category_num >= 3):
            confidence = False
        if (confidence_index[0] <= 0.6) & (category_num <= 2):
            confidence = False
        return label, confidence_index, confidence

    def return_acc(self):
        return self.acc

    def return_f1(self):
        f1 = f1_score(list(self.y), list(self.pred), average='macro')
        return float(f1)

    def print_error_case(self):
        """
        打印出预测错误的样本
        :return: None
        """
        lsts = []
        for i in range(len(set(self.y))):
            lsts.append([])
        for i, label in enumerate(self.y):
            if label != self.pred[i]:
                lsts[label].append([label, self.pred[i], self.img_names[i]])
        for i in range(len(lsts)):
            print(lsts[i])

    def print_label(self):
        print("预测值：", self.pred)
        print("真实值：", self.y)
        return self.pred, self.y

    def print_classification_report(self):
        print('\n')
        print("预测值：", self.pred)
        print("真实值：", self.y)
        print("样本量：", len(self.y))
        target_names = ['白', '淡黄',  '黄', '焦黄', '灰黑']
        try:
            print(classification_report(list(self.y), list(self.pred), target_names=target_names))
        except ValueError:
            print(classification_report(list(self.y), list(self.pred)))

    def save_classification_report(self):
        print('\n')
        print("预测值：", self.pred)
        print("真实值：", self.y)
        print(len(self.y))
        target_names = ['淡黄', '  白', '  黄', '灰黑', '焦黄']
        try:
            report = classification_report(list(self.y), list(self.pred), target_names=target_names, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('report.csv', index=True, encoding='utf-8')
        except ValueError:
            report = classification_report(list(self.y), list(self.pred), output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('report.csv', index=True, encoding='utf-8')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, nlayers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, nhead, hidden_dim),
            nlayers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x


def imshow(img, text=None, should_save=False):
    # 展示一幅tensor图像，输入是(C,H,W)
    # 将tensor转为ndarray
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10}
                 )
    # 转换为(H,W,C)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):

    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


def get_vectors(model, images):
    """

    :param model:
    :param images: 内含多个列表，子列表中存储图片的初始tensor
    :return: 输入每个图片经过模型后的向量
    """
    vectors_list = []
    model.eval()
    for i in range(len(images)):
        temp_vectors = []
        with torch.no_grad():
            for vector in images[i]:
                temp_vectors.append(model(images))
        vectors_list.append(temp_vectors)
    return vectors_list


def get_predict_label(base_vectors, vector):
    """

    :param base_vectors: 内含多个子列表，子列表中存储模型输出向量
    :param vector:
    :return:输入向量vector与base_vectors中的子列表中的向量匹配欧式距离，与哪个子列表的距离更近，就返回该子列表的索引
    """
    distances = []
    for lst in range(len(base_vectors)):
        temp_distances = []
        for vec in lst:
            temp_distances.append(F.pairwise_distance(vec, vector, keepdim=True))
        temp_distances = sorted(temp_distances)
        distances.append(temp_distances)
    label = 0
    dis = sum(distances[0][0:10])/10
    for i in range(1, len(distances)):
        if dis >= distances[i][0]:
            label = int(i)
            dis = distances[i][0]
    return label
