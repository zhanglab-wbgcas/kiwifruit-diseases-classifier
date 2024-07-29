import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pd
from tqdm import tqdm
# import time

def get_args():
    parser = argparse.ArgumentParser(description='Training script for torchvision models.')
    # parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train the model')
    # parser.add_argument('--step', type=int, default=10, help='Steps for learning rate reduction')
    # parser.add_argument('--gamma', type=float, default=0.1, help='Reduction factor for learning rate')
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

    return model, device

def validate_model(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

    average_loss = total_loss / len(loader)
    # accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None)  # None to get per-class
    recall = recall_score(all_labels, all_preds, average=None)  # None to get per-class
    f1 = f1_score(all_labels, all_preds, average=None)  # None to get per-class
    # print(average_loss)
    # print(accuracy)

    return average_loss, precision, recall, f1


def train_and_evaluate(model, device, trainloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=None, epochs=30, steps_per_epoch=len(trainloader), pct_start=0.5, anneal_strategy='cos', final_div_factor=50)

    metrics_data = { 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(30):
        model.train()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss,prec, rec, f1 = validate_model(model, device, testloader, criterion)
        # metrics_data['accuracy'].append(acc)
        metrics_data['precision'].append(prec)
        metrics_data['recall'].append(rec)
        metrics_data['f1'].append(f1)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1},  Precision: {prec}, Recall: {rec}, F1: {f1}')
        # print(metrics_data)

    return metrics_data


def plot_metrics(metrics, metric_name, save_path):
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, 31)  # There are 30 epochs
    print(len(epochs))
    print(metrics)
    print(metric_name)
    class_labels = ['Bacterial canker', 'Soft rot']  # Labels for each class
    for i, class_metric in enumerate(np.array(metrics[metric_name]).T):  # Transpose to iterate over class metrics
        
        # plt.plot(epochs, class_metric, label=f'Class {i}')
        plt.plot(epochs, class_metric, label=class_labels[i])
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/04{metric_name}.png')
    # plt.savefig(f'./twoclass/01{metric_name}.png')

    plt.close()


def main():
    trainloader, testloader = prepare_data_loaders()
    model_name='shufflenet_v2_x0_5'

    model, device = initialize_model(model_name, num_classes=2,prepth_file_path='./shufflenet_v2_x0_5.pth')    
    metrics = train_and_evaluate(model, device, trainloader, testloader)
    # for metric_name in metrics.keys():
    #     plot_metrics(metrics, metric_name, './twoclass')
    # metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = [ 'precision', 'recall', 'f1']

    for metric_name in metric_names:
        plot_metrics(metrics, metric_name, './twoclass')
    with open('04twoclass_metrics.json', 'w') as f:   #保存结果方便后期调整图片
        json.dump({k: [np.array(v).tolist() for v in metric] for k, metric in metrics.items()}, f, indent=4)

    
    

if __name__ == "__main__":
    main()
