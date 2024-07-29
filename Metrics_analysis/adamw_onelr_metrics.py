import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from thop import profile
import time
import numpy as np

#shufflenet_v2_x0_5 model 


def get_args():
    parser = argparse.ArgumentParser(description='Training script for torchvision models.')
    parser.add_argument('--model', default='shufflenet_v2_x0_5', help='Model name to import')
    # parser.add_argument('--step', type=int, default=20, help='Steps for learning rate reduction')
    parser.add_argument('--gamma', type=float, default=0.5, help='Reduction factor for learning rate')
    # parser.add_argument('--weight_decay',type=float, default=5e-4,help='weight decay')

    return parser.parse_args()


def prepare_data_loaders():
    class RobustImageFolder(ImageFolder):
        def __getitem__(self, index):
            # Attempt to load the specified item
            try:
                return super(RobustImageFolder, self).__getitem__(index)
            except Exception as e:
                print(f"Skipping image at index {index}: {e}")
                # Skip the current image and load the next one instead
                return self.__getitem__(index + 1 if index + 1 < len(self) else 0)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 稍微调整了缩放范围，减少重要特征丢失
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 减少旋转角度，减少过度变形
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 稍微减少色彩抖动的强度
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # 有条件地应用高斯模糊，模拟轻微的焦外效果
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 直接调整到网络输入尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])

    trainset =RobustImageFolder(root='./kiwi_class/train', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    testset = RobustImageFolder(root='./kiwi_class/val', transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader

def initialize_model(model_name, num_classes, prepth_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型架构但不加载预训练权重
    model = getattr(models, model_name)(pretrained=False)
    
    # 加载预训练权重
    state_dict = torch.load(prepth_file_path, map_location=device)
    model.load_state_dict(state_dict)

    # 更新分类器以匹配新的类别数
    if 'fc' in dir(model):  # 对于像ResNet这类模型
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'classifier' in dir(model):  # 对于像VGG这类模型
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    model.to(device)

    # 使用虚拟输入计算FLOPs和参数
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)

    print(f"Model: {model_name}")
    print(f"Params: {params}")
    print(f"FLOPs: {flops}")

    return model, device


def validate_model(model, device, loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():  # No gradients needed for validation, reduces memory and compute
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    val_loss = total_loss / len(loader)  # 计算平均损失
    return val_loss, precision, recall, f1, accuracy


def train_model(model, device, trainloader, testloader, args, pth_file_path, png_file_path,metrics_log):
    start_time = time.time()  # 记录训练开始时间
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=None, epochs=30, steps_per_epoch=len(trainloader), pct_start= 0.5, anneal_strategy='cos', final_div_factor=50)


    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_acc = 0  # Track the best validation accuracy
    best_val_loss = float('inf')  # Track the best validation loss
    
    for epoch in range(5):  # Consider defining epoch count in args
        model.train()
        running_loss, running_correct, total_samples = 0, 0, 0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = (running_correct.double() / total_samples).item() * 100
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        val_loss, precision, recall, f1_score, val_accuracy = validate_model(model, device, testloader,criterion)
        print("val_loss:",val_loss)
        print("precision:",precision)
        print("recall:",recall)
        print("f1_score:",f1_score)
        print("val_accuracy:",val_accuracy)
        # val_loss.append(val_epoch_loss)
        # val_acc.append(val_epoch_acc)

        scheduler.step()  # 更新学习率


        metrics_log.append((val_loss, precision, recall, f1_score, val_accuracy))

def main():

    args = get_args()
    trainloader, testloader = prepare_data_loaders()
    
    pth_save_directory = './pth_save'
    pic_save_directory = './pic_save'
    os.makedirs(pth_save_directory, exist_ok=True)
    os.makedirs(pic_save_directory, exist_ok=True)
    pth_file_path = os.path.join(pth_save_directory, f"best_{args.model}_{args.gamma}_onelr_adamw_model.pth")
    png_file_path = os.path.join(pic_save_directory, f"{args.model}_{args.gamma}_onelr_adamw_loss_acc.png")
    # losses = []  # 用于存储每次训练的最佳验证损失
    metrics_log = []
    for i in range(10):  # 训练模型10次
        
        model, device = initialize_model(args.model, num_classes=2, prepth_file_path='./shufflenet_v2_x0_5.pth')
        train_model(model, device, trainloader, testloader, args, pth_file_path, png_file_path,metrics_log)

        # losses.append(best_loss)
        # print(f'Training {i+1}: Best Validation Loss: {best_loss:.4f}')
    # 在训练循环外部
    average_precision = np.mean([m[1] for m in metrics_log])
    average_recall = np.mean([m[2] for m in metrics_log])
    average_f1_score = np.mean([m[3] for m in metrics_log])
    average_accuracy = np.mean([m[4] for m in metrics_log])

    print(f'Average Precision: {average_precision:.2f}')
    print(f'Average Recall: {average_recall:.2f}')
    print(f'Average F1 Score: {average_f1_score:.2f}')
    print(f'Average Accuracy: {average_accuracy:.2f}')


    

if __name__ == "__main__":
    main()
