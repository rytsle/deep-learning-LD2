import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


class DataFrameDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Assuming the 'Image' column contains filenames or prefixes
        img_val = str(self.df.iloc[idx]['Image'])
        if not img_val.endswith('.png') and not img_val.endswith('.jpg'):
            img_val += '.png'
            
        img_path = os.path.join(self.img_dir, img_val)
            
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = int(self.df.iloc[idx]['class_label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model(model, model_name, train_df, val_df, img_dir='LD2_dataset/images', epochs=10, batch_size=32, lr=0.001, device=None):
    if device is None:
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    best_weights_path = f"model_params/{model_name}_best_weights.pth"
    history_path = f"model_params/{model_name}_training_history.json"
 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generic transforms
    transform_train = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = DataFrameDataset(train_df, img_dir, transform=transform_train)
    val_dataset = DataFrameDataset(val_df, img_dir, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = -1.0
    best_epoch = 1
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                loop.set_postfix(loss=loss.item())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), best_weights_path)
            saved_msg = f" -> Saved new best weights to {best_weights_path} (Val F1: {epoch_val_f1:.4f})"
            best_epoch = epoch + 1
        else:
            saved_msg = ""
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)
        history['best_epoch'] = best_epoch
        print(f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | F1 Macro: {epoch_train_f1:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | F1 Macro: {epoch_val_f1:.4f}{saved_msg}\n")
        

    with open(f'{history_path}', 'w') as f:
        json.dump(history, f, indent=4)
    return model, history



def test_model(model, model_name, metrics_save_dir=None, plots_save_dir=None, test_df=None, img_dir='LD2_dataset/images', device=None, batch_size=32):
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_test = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Reuse DataFrameDataset for loading test data
    test_dataset = DataFrameDataset(test_df, img_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
            
    # Evaluate and compare to test_df values
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Class-level metrics like classification_report
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision,
        'recall_macro': recall,
        'classification_report': class_report
    }
    
    if metrics_save_dir is not None:
        metrics_path = os.path.join(metrics_save_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
    # Create the confusion matrix and save as PNG
    if plots_save_dir is not None:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = os.path.join(plots_save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
    
    return metrics



def load_trained_model(model_class, weights_path, num_classes=3, device=None):
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the model architecture
    model = model_class(num_classes=num_classes)
    
    # Load the state dictionary
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"Model {model_class.__name__} loaded successfully from {weights_path}")
    return model

