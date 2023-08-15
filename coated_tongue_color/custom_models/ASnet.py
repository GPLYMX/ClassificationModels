import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet152, resnet34

from utils import Flatten


class ASnet(nn.Module):

    def __init__(self):
        super(ASnet, self).__init__()
        self.trained_model = vgg16(pretrained=True)
        self.modelA = nn.Sequential(*list(self.trained_model.children())[0][0:17],
                                    nn.Conv2d(256, 64, kernel_size=(1, 1)),
                                    nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                                    Flatten(),
                                    nn.Linear(in_features=3136, out_features=500, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(500, 5),
                                    nn.ReLU(inplace=True)
                                    )
        self.modelS = nn.Sequential(*list(self.trained_model.children())[0][0:23],
                                    nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, ceil_mode=False),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
                                    nn.ConvTranspose2d(in_channels=512, out_channels=256, padding=2, kernel_size=8,
                                                       stride=2),
                                    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                                    Flatten(),
                                    nn.Linear(in_features=12544, out_features=5000, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(5000, 1000),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(1000, 200),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(200, 5)
                                    )
        self.vgg = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                 Flatten(),
                                 nn.Linear(25088, 5000),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(5000, 200),
                                 nn.Dropout(p=0.3),
                                 nn.ReLU(),
                                 nn.Linear(200, 5)
                                 )
        self.model3 = nn.Sequential(
            nn.Linear(30, 5)
        )
        self.trained_model2 = resnet34(pretrained=True)  # .to(device)
        self.res34 = nn.Sequential(*list(self.trained_model2.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                   Flatten(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(512, 128),
                                   nn.Dropout(p=0.3),
                                   nn.ReLU(),
                                   nn.Linear(128, 15)
                                   )

    def forward(self, input1):
        x1 = self.modelA(input1)
        x2 = self.modelS(input1)
        x3 = self.vgg(input1)
        x4 = self.res34(input1)
        output1 = torch.cat([x1, x2, x3, x4], dim=1)
        output1 = self.model3(output1)
        return output1


if __name__ == '__main__':
    VGG16 = vgg16()
    print(VGG16)

# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): ReLU(inplace=True)
#     (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (14): ReLU(inplace=True)
#     (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace=True)
#     (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): ReLU(inplace=True)
#     (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

