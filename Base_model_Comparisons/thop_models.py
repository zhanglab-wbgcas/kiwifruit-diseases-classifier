import torch
from torchvision import models
from thop import profile

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
# model = models.mobilenet_v2(pretrained=True).to(device)
model = models.squeezenet1_0(pretrained=True).to(device)

# 设置输入张量
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# 计算FLOPs和Params
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)

# 打印结果
print(f"Total FLOPs: {flops}")
print(f"Total Parameters: {params}")