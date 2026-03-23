import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_resnet18(pretrained: bool, num_classes: int = 4) -> nn.Module:
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model


def train_model_resnet(model, model_name, train_loader, test_loader, epochs, lr, device, save_weights, weights_save_dir):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_f1': [], 'test_f1': [], 'best_epoch': 1}
    best_test_f1 = -1.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        all_train_preds = []
        all_train_labels = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train] {model_name}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        # Evaluation phase
        model.eval()
        all_test_preds = []
        all_test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)

        history['train_f1'].append(train_f1)
        history['test_f1'].append(test_f1)

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            history['best_epoch'] = epoch + 1
            if save_weights:
                os.makedirs(weights_save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(weights_save_dir, f'{model_name}_best_weights.pth'))

        print(f"Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")

    return model, history


def test_model_resnet(model, model_name, test_loader, device, metrics_save_dir):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision,
        'recall_macro': recall,
        'classification_report': class_report
    }

    os.makedirs(metrics_save_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_save_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics
