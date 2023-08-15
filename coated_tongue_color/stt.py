import torch
from torchvision.models import swin_transformer

print(type(swin_transformer))
a = torch.rand(2, 3, 224, 224)
model = swin_transformer()
b = model(a)
print(b.shape)

