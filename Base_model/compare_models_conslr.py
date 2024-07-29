


import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
import json

import time


def get_args():
    parser = argparse.ArgumentParser(description='Training script for various torchvision models.')
    return parser.parse_args()

def prepare_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = datasets.ImageFolder(root='./kiwi_class/train', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testset = datasets.ImageFolder(root='./kiwi_class/val', transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

def initialize_model(model_name, num_classes, prepth_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_name)(pretrained=False)
    state_dict = torch.load(prepth_file_path, map_location=device)
    model.load_state_dict(state_dict)
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
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


def train_model(model, device, trainloader, testloader, optimizer, pth_file_path):
    start_time = time.time()  # 记录训练开始时间
    
    criterion = nn.CrossEntropyLoss()
    

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
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
    

    trainloader, testloader = prepare_data_loaders()
    all_val_losses = {}
    all_val_accuracies = {}
    all_train_losses = {}
    all_train_accuracies = {}
    results = {}
    pth_save_directory = './pth_save'

    models_to_test = {
        'alexnet': './01multi_weight_pro/pth_dir/alexnet.pth',
        'vgg19': './01multi_weight_pro/pth_dir/vgg19.pth',
        'resnet18': './01multi_weight_pro/pth_dir/resnet18.pth',
        'efficientnet_b0': './01multi_weight_pro/pth_dir/efficientnet_b0.pth',
        'mobilenet_v2': './01multi_weight_pro/pth_dir/mobilenet_v2.pth',
        'mobilenet_v3_small': './01multi_weight_pro/pth_dir/mobilenet_v3_small.pth',
        'mnasnet0_5': './01multi_weight_pro/pth_dir/mnasnet0_5.pth',
        'shufflenet_v2_x0_5': './01multi_weight_pro/pth_dir/shufflenet_v2_x0_5.pth'
    }

    for model_name, path in models_to_test.items():
        model, device = initialize_model(model_name, 2, path)
        print('model_name:',model_name)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        print(f'Training {optimizer}...')



        pth_file_path = os.path.join(pth_save_directory, f"best_model_name_{model_name}_model.pth")
        val_losses, val_accuracies,train_losses,train_accuracies = train_model(model, device, trainloader, testloader, optimizer, pth_file_path)
        all_val_losses[model_name] = val_losses
        all_val_accuracies[model_name] = val_accuracies
        all_train_losses[model_name] = train_losses
        all_train_accuracies[model_name] = train_accuracies
        results[f"config_{model_name}"] = {'val_loss': val_losses, 'val_accuracy': val_accuracies, 'train_loss': train_losses, 'train_accuracy': train_accuracies}
    with open('compare_models_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_and_save_metrics(all_val_losses, 'Validation Loss Comparison', 'Loss', './compare_pic/models_val_loss_comparison.png')
    plot_and_save_metrics(all_val_accuracies, 'Validation Accuracy Comparison', 'Accuracy (%)', './compare_pic/models_val_accuracy_comparison.png')
    plot_and_save_metrics(all_train_losses, 'Training Loss Comparison', 'Loss', './compare_pic/models_train_loss_comparison.png')
    plot_and_save_metrics(all_train_accuracies, 'Training Accuracy Comparison', 'Accuracy (%)', './compare_pic/models_train_accuracy_comparison.png')




if __name__ == "__main__":
    main()


