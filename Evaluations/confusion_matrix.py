import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_model(num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.shufflenet_v2_x0_5(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    return model, device

def load_model_weights(model, path):
    state_dict = torch.load(path)
    keys_to_delete = [key for key in state_dict if "total_ops" in key or "total_params" in key]
    for key in keys_to_delete:
        del state_dict[key]
    model.load_state_dict(state_dict)
    return model

def prepare_data_loader():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testset = datasets.ImageFolder(root='./kiwi_class/test', transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return testloader

def evaluate_model(model, device, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # print(all_preds)
            # print(all_labels)
    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    print('cm',cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,annot_kws={"color":"black"})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./twoclass/confusion_matrix.png')
    plt.close()

def main():
    model, device = initialize_model(num_classes=2)
    model = load_model_weights(model, 'best_shufflenet_v2_x0_5_model.pth')
    testloader = prepare_data_loader()
    labels, preds = evaluate_model(model, device, testloader)
    classes = testloader.dataset.classes
    plot_confusion_matrix(labels, preds, classes)

if __name__ == "__main__":
    main()
