import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from thop import profile
import time
import json


def get_args():
    parser = argparse.ArgumentParser(description='Training script for torchvision models.')
    # parser.add_argument('--model', default='resnet18', help='Model name to import')
    parser.add_argument('--step', type=int, default=10, help='Steps for learning rate reduction')
    parser.add_argument('--gamma', type=float, default=0.1, help='Reduction factor for learning rate')
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


def plot_and_save_metrics(metrics, title, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    for combine, data in metrics.items():
        plt.plot(data, label=combine)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def validate_model(model, device, loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed for validation, reduces memory and compute
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100 * total_correct / total
    return average_loss, accuracy


def train_model(model, device, trainloader, testloader, optimizer, scheduler,pth_file_path):
    start_time = time.time()  # 记录训练开始时间
    
    criterion = nn.CrossEntropyLoss()
    

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    # best_val_acc = 0  # Track the best validation accuracy
    best_val_loss = float('inf')  # Track the best validation loss

    for epoch in range(30):  # Consider defining epoch count in args
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

        val_epoch_loss, val_epoch_acc = validate_model(model, device, testloader, criterion)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Check the type of scheduler and apply appropriate step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_epoch_loss)  # For ReduceLROnPlateau, pass in validation loss
        else:
            scheduler.step()


        print(f'Epoch [{epoch+1}/30], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')

        # if val_epoch_acc > best_val_acc:
        #     best_val_acc = val_epoch_acc
        # #     torch.save(model.state_dict(), pth_file_path)
        # #     print(f'New best model saved with accuracy: {best_val_acc:.2f}%')
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), pth_file_path)
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    end_time = time.time()  # 记录训练结束时间
    total_time = end_time - start_time  # 计算总训练时间
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = (total_time % 3600) % 60
    print('total_time:',total_time)
    print(f'Training completed in {hours:.0f} hours, {minutes:.0f} minutes and {seconds:.0f} seconds.')
    # plot_metrics(train_loss, train_acc, val_loss, val_acc, png_file_path)
    print(f'Lowest Validation Loss: {best_val_loss:.4f}')
    return val_loss, val_acc,train_loss,train_acc

def main():

    args = get_args()
    model_name='shufflenet_v2_x0_5'
    trainloader, testloader = prepare_data_loaders()
    all_val_losses = {}
    all_val_accuracies = {}
    all_train_losses = {}
    all_train_accuracies = {}
    results = {}
    pth_save_directory = './pth_save'

    for i in range(3):  # Assuming three different configurations
        model, device = initialize_model(model_name, num_classes=2,prepth_file_path='./shufflenet_v2_x0_5.pth')
        if i == 0:
            optimizer = optim.AdamW(model.parameters(), lr=0.01)
            scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=None, epochs=30, steps_per_epoch=len(trainloader), pct_start=0.5, anneal_strategy='cos', final_div_factor=50)
        elif i == 1:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.gamma, patience=args.step, verbose=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, nesterov=True)
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.gamma, patience=args.step, verbose=True)
        lr=0.01
        # print(f'Training {optimizer}...')
        pth_file_path = os.path.join(pth_save_directory, f"best_model_name_{args.gamma}_{args.step}_{i}_{lr}_model.pth")
        val_losses, val_accuracies,train_losses,train_accuracies = train_model(model, device, trainloader, testloader, optimizer, scheduler,pth_file_path)
        all_val_losses[i] = val_losses
        all_val_accuracies[i] = val_accuracies
        all_train_losses[i] = train_losses
        all_train_accuracies[i] = train_accuracies
        results[f"config_{i}"] = {'val_loss': val_losses, 'val_accuracy': val_accuracies, 'train_loss': train_losses, 'train_accuracy': train_accuracies}
    with open('shufflecompare_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_and_save_metrics(all_val_losses, 'Validation Loss Comparison', 'Loss', './compare_pic/shuffle_val_loss_comparison.png')
    plot_and_save_metrics(all_val_accuracies, 'Validation Accuracy Comparison', 'Accuracy (%)', './compare_pic/shuffle_val_accuracy_comparison.png')
    plot_and_save_metrics(all_train_losses, 'Training Loss Comparison', 'Loss', './compare_pic/shuffle_train_loss_comparison.png')
    plot_and_save_metrics(all_train_accuracies, 'Training Accuracy Comparison', 'Accuracy (%)', './compare_pic/shuffle_train_accuracy_comparison.png')




if __name__ == "__main__":
    main()
