import torch
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import vgg11
from torchvision.models import vgg16
from torchvision.models import vgg16_bn
from torchvision.models import vgg19
from torchvision.models import vgg19_bn
from torchvision.models import vgg13
from torchvision.models import densenet121
from torchvision.models import densenet201
from torchvision.models import resnet34
from torchvision.models import densenet161
from torchvision.models import googlenet
import torch.nn as nn
from torchsummary import summary

from utils import Flatten
# import ssl


class Resnet152(nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        # ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = resnet152(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 5),
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        # ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = resnet101(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 5),
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        # ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = resnet50(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(2048, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 5),
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.trained_model = resnet34(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.trained_model = resnet18(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.trained_model = vgg11(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.trained_model = vgg13(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.trained_model = vgg16(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG16_bn(nn.Module):
    def __init__(self):
        super(VGG16_bn, self).__init__()
        self.trained_model = vgg16_bn(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.trained_model = vgg19(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class VGG19_bn(nn.Module):

    def __init__(self):
        super(VGG19_bn, self).__init__()
        self.trained_model = vgg19_bn(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(25088, 5000),
                                    nn.ReLU(),
                                    nn.Linear(5000, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = densenet121(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(50176, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Densenet161(nn.Module):
    def __init__(self):
        super(Densenet161, self).__init__()
        # ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = densenet161(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(50176, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class Densenet201(nn.Module):
    def __init__(self):
        super(Densenet201, self).__init__()
        ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = densenet201(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(1920, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        # ssl._create_default_https_context = ssl._create_unverified_context
        self.trained_model = googlenet(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(1024, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 5)
                                    )

    def forward(self, input1):
        output1 = self.model1(input1)
        return output1


if __name__ == '__main__':
    a = vgg16()
    for name in a.children():
        print('dsf')
        print(name)
    # b = torch.randn(10, 3, 32, 32)
    # print(a(b))
    a = a.cuda()
    print(summary(a, (3, 50, 50)))
    # print(a)
    print(*list(a.children())[-2:])