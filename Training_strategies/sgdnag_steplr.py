import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from thop import profile
import time



def get_args():
    parser = argparse.ArgumentParser(description='Training script for torchvision models.')
    parser.add_argument('--model', default='resnet18', help='Model name to import')
    parser.add_argument('--step', type=int, default=20, help='Steps for learning rate reduction')
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

def plot_metrics(train_loss, train_acc, val_loss, val_acc, save_path):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss over epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Accuracy over epochs')
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


def train_model(model, device, trainloader, testloader, args, pth_file_path, png_file_path):
    start_time = time.time()  # 记录训练开始时间
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, dampening=0,nesterov=True)
    scheduler = StepLR(optimizer, step_size=args.step,gamma=args.gamma)  # 每5个epoch，学习率乘以0.5

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_acc = 0  # Track the best validation accuracy
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

        scheduler.step()
        print(f'Epoch [{epoch+1}/30], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')


        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
        #     torch.save(model.state_dict(), pth_file_path)
        #     print(f'New best model saved with accuracy: {best_val_acc:.2f}%')
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            # torch.save(model.state_dict(), pth_file_path)
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
    return best_val_loss

def main():

    args = get_args()
    trainloader, testloader = prepare_data_loaders()
    # model, device = initialize_model(args.model, num_classes=2,prepth_file_path='./shufflenet_v2_x0_5.pth')
    
    pth_save_directory = './pth_save'
    pic_save_directory = './pic_save'
    os.makedirs(pth_save_directory, exist_ok=True)
    os.makedirs(pic_save_directory, exist_ok=True)
    pth_file_path = os.path.join(pth_save_directory, f"best_{args.model}_{args.step}_{args.gamma}_steplr_sgdnag_model.pth")
    png_file_path = os.path.join(pic_save_directory, f"{args.model}_{args.step}_{args.gamma}_steplr_sgdnag_loss_acc.png")
    
    losses = []  # 用于存储每次训练的最佳验证损失

    for i in range(10):  # 训练模型10次
        model, device = initialize_model(args.model, num_classes=2, prepth_file_path='./shufflenet_v2_x0_5.pth')
        best_loss = train_model(model, device, trainloader, testloader, args, pth_file_path, png_file_path)

        losses.append(best_loss)
        print(f'Training {i+1}: Best Validation Loss: {best_loss:.4f}')

    # 计算平均最佳验证损失
    average_loss = sum(losses) / len(losses)
    print(f'Average Best Validation Loss: {average_loss:.4f}')


if __name__ == "__main__":
    main()
