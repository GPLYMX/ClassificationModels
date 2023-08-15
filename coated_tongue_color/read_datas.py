# coding=utf-8
import json
import os
from PIL import Image
import re
import random
from shutil import copyfile

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2 as cv
import pandas as pd


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img


class ReaderDatas(Dataset):

    def __init__(self, picture_root='.\\datas\\all_seg_crop', resize=244, mode="train",
                 task_label="苔色", category=['白', '淡黄',  '黄', '焦黄', '灰黑'], label_filename='.\\datas\\detail_top1.json',
                 data_type=1):
        """

        :param picture_root: 图片的地址
        :param resize: 神经网络输入维度
        :param mode: 数据集的作用
        :param task_label: 任务类型
        :param category: 本任务包含的具体分类
        :param label_filename: 存储标签信息的json文件路径
        """
        super(ReaderDatas, self).__init__()

        self.picture_root = picture_root
        self.label_filename = label_filename
        self.task_label = task_label
        self.category = category
        self.category_num = len(category)
        self.resize = resize
        # ['.\\datas\\all_seg_crop\\301.png', '.\\datas\\all_seg_crop\\704.png', ]
        self.images = []
        self.mode = mode
        # [1, 2, 0, 1, 0, 0]
        self.labels = []
        self.confidences = []
        # 多个列表，每个列表中为某一类的图片地址
        self.images_temp = []
        self.labels_temp = []
        self.confidences_coe = []
        # 读取的数据格式
        self.data_type = data_type
        # 获取当前的工作目录
        self.abspath = os.path.dirname(os.path.abspath("__file__"))

        # 根据种类数量，创建相应个数的self.images和self.label
        for i in range(len(self.category)):
            self.images_temp.append([])
            self.labels_temp.append([])
            self.confidences_coe.append([])

        with open(label_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for img in json_data.keys():
                try:
                    label, confidence_index, confidence = self.extract_label2(json_data[img][self.task_label])
                except TypeError:
                    label, confidence_index, confidence = self.extract_label2(json_data[img][self.task_label])
                except Exception as E:
                    pass
                if confidence:
                    if img in os.listdir(picture_root):
                        self.images_temp[label].append(os.path.join(self.picture_root, img))
                        self.labels_temp[label].append(label)
                        self.confidences_coe[label].append(confidence_index)
                    elif img + '.png' in os.listdir(picture_root):
                        self.images_temp[label].append(os.path.join(self.picture_root, img + '.png'))
                        self.labels_temp[label].append(label)
                        self.confidences_coe[label].append(confidence_index)
        # 每一类分到各自的文件夹中
        self.datas_classify()

        # # 打散
        # rand_num = random.randint(0, 100)
        # random.seed(rand_num)
        # random.shuffle(self.images)
        # random.seed(rand_num)
        # random.shuffle(self.labels)

        if mode == 'train':
            for i in range(len(self.category)):
                self.images += self.images_temp[i][:int(0.7 * len(self.images_temp[i]))]
                self.labels += self.labels_temp[i][:int(0.7 * len(self.labels_temp[i]))]
        elif mode == 'val':
            for i in range(len(self.category)):
                self.images += self.images_temp[i][int(0.6 * len(self.images_temp[i])):int(0.8 * len(self.images_temp[i]))]
                self.labels += self.labels_temp[i][int(0.6 * len(self.labels_temp[i])):int(0.8 * len(self.labels_temp[i]))]
        elif mode == 'test':
            for i in range(len(self.category)):
                self.images += self.images_temp[i][int(0.7 * len(self.images_temp[i])):]
                self.labels += self.labels_temp[i][int(0.7 * len(self.labels_temp[i])):]
        else:
            print('请重新输入mode')

    def datas_classify(self):
        """
        为每一类的图片创建各自的文件夹、并复制到该文件夹中
        :return:
        """

        datas_root = os.path.abspath(os.path.join(self.picture_root, os.path.pardir))
        for i in range(len(self.category)):
            if not os.path.exists(os.path.join(datas_root, "category", str(i))):
                os.makedirs(os.path.join(datas_root, "category", str(i)))
                for img in self.images_temp[i]:
                    (p, filename) = os.path.split(img)
                    copyfile(img, os.path.join(datas_root, "category", str(i), filename))

    def extract_label1(self, string):
        """
        从json文件中的标签中提取相应的信息
        :param string:
        :return:
        """
        confidence = True
        label = re.match("[\u4E00-\u9FA5]+", string).group()
        try:
            label = int(self.category.index(label))
        except ValueError:
            label = int(0)

        confidence_index = re.search(r'\d\.\d*', string).group()
        confidence_index = float(confidence_index)

        if (confidence_index < 0.5) & (self.category_num >= 3):
            confidence = False
        if (confidence_index < 0.6) & (self.category_num == 2):
            confidence = False

        return label, confidence_index, confidence

    def extract_label2(self, string):
        """
        从新的json文件中的标签提取相应信息
        :param string:
        :return:
        """
        confidence = True
        vote_num = list(string.values())
        label = vote_num.index(max(vote_num))
        confidence_index = max(vote_num)/sum(vote_num)

        # # 新数据与老数据的json格式不一致
        # if label == 0:
        #     label = 1
        # elif label == 1:
        #     label = 0
        # elif label == 3:
        #     label = 4
        # elif label == 4:
        #     label = 3

        if (confidence_index <= 0.6) & (self.category_num >= 3):
            confidence = False
        if (confidence_index <= 0.6) & (self.category_num <= 2):
            confidence = False

        return label, confidence_index, confidence

    def __len__(self):
        return len(self.images)

    # def denormalize(self, x_hat):
    #
    #     mean = [0.435, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    #     std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    #
    #     x = x_hat * std + mean
    #
    #     return x

    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]
        if self.mode != 'test':
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                lambda x:AddGaussianNoise(mean=random.uniform(0.5, 1.5),
                                          variance=0.5, amplitude=random.uniform(0, 45))(x)
            ])
        else:
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        img = tf(img)
        label = torch.tensor(int(label))
        return img, label


class ReaderClassfiedData(Dataset):
    """
    用于读取划分好类别的数据
    """

    def __init__(self, root='datas/data3/category', mode='train',
                 resize=224, random_seed=1):
        super(ReaderClassfiedData, self).__init__()
        self.resize = resize
        self.random_seed = random_seed
        self.mode = mode
        self.root = root
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        self.test_images = []
        self.test_labels = []
        self.labels = []
        self.category_num = len(os.listdir(root))
        self.create_label()
        self.tf1 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.RandomRotation(random.uniform(10, 180)),
            transforms.RandomRotation(random.uniform(0, 359)),
            transforms.RandomAffine(degrees=0, fill=(0, 0, 0), shear=random.uniform(0, 20)),
            transforms.RandomRotation(random.uniform(0, 359)),
            # transforms.RandomRotation(90),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            # 通用值
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # 舌苔值
            # transforms.Normalize(mean=[0.3798, 0.2742, 0.2732],
            #                      std=[0.3175, 0.2436, 0.2463])
            # lambda x: Image.open(x).convert('RGB'),
            # # lambda x: self.sep(x),
            # transforms.Resize((self.resize, self.resize)),
            # lambda x: AddGaussianNoise(mean=random.uniform(0.5, 1.5),
            #                            variance=0.5, amplitude=random.uniform(0, 45))(x),
            # transforms.RandomRotation(random.uniform(0, 359)),
            # transforms.CenterCrop(0.8*self.resize),
            # # transforms.FiveCrop(0.7*self.resize),
            # transforms.RandomRotation(random.uniform(0, 359)),
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            # transforms.RandomRotation(random.uniform(0, 359)),
            # transforms.RandomCrop(random.uniform(0.6, 1)*self.resize),
            # transforms.RandomRotation(random.uniform(0, 359)),
            # transforms.RandomAffine(degrees=0, translate=(0.2, 0.5), fill=(0, 125, 0)),
            # # transforms.TenCrop(0.8*self.resize, vertical_flip=True),
            # transforms.Resize((self.resize, self.resize)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.435, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.tf2 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            # lambda x: self.sep(x),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def sep(self, image):
        img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        mask = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)
        mask1 = mask.reshape((-1, 1))
        mask1[mask1 != 0] = 1
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)

        a = a.reshape((-1, 1))
        z = a * mask1
        idx = np.flatnonzero(z)
        a = pd.DataFrame(z).replace(0, np.NAN)
        a.dropna(inplace=True)
        a = np.float32(a)
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        ret, label, center = cv.kmeans(a, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        cmax = np.min(center)
        res = center[label.flatten()]
        res2 = np.zeros_like(mask1)
        res2[idx] = res
        res2[res2 != cmax] = 0
        res2[res2 == cmax] = 1
        res2 = res2.reshape((mask.shape))
        coat = cv.merge([res2, res2, res2])
        sub = 1 - coat

        coats = img * coat
        subs = img * sub

        coats = Image.fromarray(cv.cvtColor(coats, cv.COLOR_BGR2RGB))
        return coats

    def create_label(self):
        for dirs in os.listdir(self.root):
            temp_images = []
            temp_labels = []
            for img in os.listdir(os.path.join(self.root, dirs)):
                temp_images.append(os.path.join(self.root, dirs, img))
                temp_labels.append(int(dirs))
            random.seed(self.random_seed)
            random.shuffle(temp_images)
            random.seed(self.random_seed)
            random.shuffle(temp_labels)
            self.train_images += temp_images[:int(0.7 * len(temp_images))]
            self.train_labels += temp_labels[:int(0.7 * len(temp_labels))]
            self.val_images += temp_images[int(0.7 * len(temp_images)):int(0.9 * len(temp_images))]
            self.val_labels += temp_labels[int(0.7 * len(temp_labels)):int(0.9 * len(temp_labels))]
            self.test_images += temp_images[int(0.7 * len(temp_images)):]
            self.test_labels += temp_labels[int(0.7 * len(temp_labels)):]
        random.seed(self.random_seed)
        random.shuffle(self.train_images)
        random.seed(self.random_seed)
        random.shuffle(self.train_labels)

        if self.mode == "train":
            self.labels = self.train_labels
        elif self.mode == 'val':
            self.labels = self.val_labels
        else:
            self.labels = self.test_labels

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images)
        elif self.mode == 'val':
            return len(self.val_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img, label = self.train_images[idx], self.train_labels[idx]
            img_name = os.path.basename(img)
            img = self.tf1(img)
        elif self.mode == 'val':
            img, label = self.val_images[idx], self.val_labels[idx]
            img_name = os.path.basename(img)
            img = self.tf2(img)
        else:
            img, label = self.test_images[idx], self.test_labels[idx]
            img_name = os.path.basename(img)
            img = self.tf2(img)
        label = torch.tensor(int(label))
        return img, label, img_name


class ReaderClassfiedCoatData(ReaderClassfiedData):
    def __init__(self, root='datas/data21/category', mode='train',
                 resize=224, random_seed=3):
        super(ReaderClassfiedCoatData, self).__init__(root, mode, resize, random_seed)
        self.tf1 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            lambda x: self.sep(x),
            transforms.Resize((self.resize, self.resize)),
            lambda x: AddGaussianNoise(mean=random.uniform(0.5, 1.5),
                                       variance=0.5, amplitude=random.uniform(0, 45))(x),
            transforms.RandomRotation(random.uniform(0, 359)),
            transforms.CenterCrop(0.8 * self.resize),
            # transforms.FiveCrop(0.7*self.resize),
            transforms.RandomRotation(random.uniform(0, 359)),
            transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomRotation(random.uniform(0, 359)),
            transforms.RandomCrop(random.uniform(0.6, 1) * self.resize),
            transforms.RandomRotation(random.uniform(0, 359)),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.5), fill=(0, 125, 0)),
            # transforms.TenCrop(0.8*self.resize, vertical_flip=True),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def sep(self, image):
        img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        mask = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)
        mask1 = mask.reshape((-1, 1))
        mask1[mask1 != 0] = 1
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)

        a = a.reshape((-1, 1))
        z = a * mask1
        idx = np.flatnonzero(z)
        a = pd.DataFrame(z).replace(0, np.NAN)
        a.dropna(inplace=True)
        a = np.float32(a)
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        ret, label, center = cv.kmeans(a, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        cmax = np.min(center)
        res = center[label.flatten()]
        res2 = np.zeros_like(mask1)
        res2[idx] = res
        res2[res2 != cmax] = 0
        res2[res2 == cmax] = 1
        res2 = res2.reshape((mask.shape))
        coat = cv.merge([res2, res2, res2])
        sub = 1 - coat

        coats = img * coat
        subs = img * sub

        coats = Image.fromarray(cv.cvtColor(coats, cv.COLOR_BGR2RGB))
        return coats

    def __getitem__(self, idx):
        if self.mode == 'train':
            img, label = self.train_images[idx], self.train_labels[idx]
            img_name = os.path.basename(img)
            img = self.tf1(img)
        else:
            img, label = self.test_images[idx], self.test_labels[idx]
            img_name = os.path.basename(img)
            img = self.tf2(img)
        label = torch.tensor(int(label))
        return img, label, img_name

     
class ReaderSiameseDatas(Dataset):
    """
    用于siamese网络的数据读取
    """
    def __init__(self, picture_root='..\\datas\\category', mode="train", resize=244):
        super(ReaderSiameseDatas, self).__init__()
        self.picture_root = picture_root
        self.resize = resize
        self.mode = mode
        # self.images中存储图片对(路径)，用于训练
        self.images = []
        # self.labels存储self.images的图片对标签，只含0、1
        self.labels = []
        # 用于测试
        self.test_images = []
        self.test_labels = []
        # 用于测试的基准库
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
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    def make_tag(self, ratio=0.8, sample_num=2000):

        for i in range(self.category_num):
            temp_list = []
            for j in os.listdir(os.path.join(self.picture_root, str(i))):
                temp_list.append(str(os.path.join(self.picture_root, str(i), j)))
            temp_list = sorted(temp_list)
            self.test_images += temp_list[int(ratio*len(temp_list)):]
            self.base_images += temp_list[0:int(ratio*len(temp_list))]
            for length in range(len(temp_list[int(ratio*len(temp_list)):])):
                self.test_labels.append(str(i))
            for length in range(len(temp_list[0:int(ratio*len(temp_list))])):
                self.base_labels.append(str(i))
            self.temp_images.append(temp_list[0:int(ratio*len(temp_list))])

        # 增加同类图片对和标签
        for i in range(sample_num):
            index = random.randint(0, len(self.temp_images)-1)
            self.images.append(random.sample(self.temp_images[index], 2))
            self.labels.append(int(0))
        # 增加非同类图片对和标签
        for i in range(sample_num):
            index = random.sample(range(0, len(self.temp_images)), 2)
            self.images.append([random.choice(self.temp_images[index[0]]), random.choice(self.temp_images[index[1]])])
            self.labels.append(int(1))

        # 打散
        rand_num = random.randint(0, 100)
        random.seed(rand_num)
        random.shuffle(self.images)
        random.seed(rand_num)
        random.shuffle(self.labels)

    def get_base_tensors(self):
        base_tensors = []
        for lst in self.temp_images:
            temp_tensors = []
            for img in lst:
                img = self.tf(img)
                temp_tensors.append(img)
            base_tensors.append(temp_tensors)
        return base_tensors

    def get_test_tensors(self):
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


if __name__ == '__main__':

    a = ReaderDatas(picture_root='.\\datas\\data1\\all_seg_crop', resize=244, mode="train",
                    task_label="苔色", category=['白', '淡黄', '黄', '焦黄', '灰黑'], label_filename='.\\datas\\data1\\second_total_detail(20220929)_sm.json', data_type=2)
    a.datas_classify()
