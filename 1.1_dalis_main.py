import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from helpers_1_dalis.architekturos import KetvirtaArchitektura, PenktaArchitektura, AstuntaArchitektura, IndividualiArchitektura
from helpers_1_dalis.train_evaluate import train_model, test_model, load_trained_model
from helpers_1_dalis.visualize import visualize_history

def main():
    # 1. Duomenų nuskaitymas
    print("Loading data...")
    labels_data = pd.read_csv('LD2_dataset/labels.csv', dtype={'Image': str, 'class_label': int})
    
    # Remove rows with class_label 5 or 8
    labels_data = labels_data[~labels_data['class_label'].isin([5, 8])]

    # Remap class labels: (0,2,4)->0, (3,6)->1, (1,7,9)->2
    label_map = {0: 0, 2: 0, 4: 0, 3: 1, 6: 1, 1: 2, 7: 2, 9: 2}
    labels_data['class_label'] = labels_data['class_label'].map(label_map)

    # Train, val, test split
    train_val_df, test_df = train_test_split(labels_data, test_size=0.1, stratify=labels_data['class_label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.111, stratify=train_val_df['class_label'], random_state=42)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Model parameters
    num_classes = 3
    epochs = 20
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models_to_run = [
        (KetvirtaArchitektura, "KetvirtaArchitektura"),
        (PenktaArchitektura, "PenktaArchitektura"),
        (AstuntaArchitektura, "AstuntaArchitektura"),
        (IndividualiArchitektura, "IndividualiArchitektura")
    ]
    
    os.makedirs('model_params', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    for ModelClass, model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Starting pipeline for {model_name}")
        print(f"{'='*50}")
        
        # 1. Create Model
        model = ModelClass(num_classes=num_classes)
        
        # 2. Train Model
        print(f"--- Training {model_name} ---")
        _, history = train_model(
            model=model, 
            model_name=model_name, 
            train_df=train_df, 
            val_df=val_df, 
            img_dir='LD2_dataset/images', 
            epochs=epochs, 
            batch_size=32, 
            lr=0.001, 
            device=device
        )
        
        # 3. Visualize Training
        print(f"--- Visualizing Training for {model_name} ---")
        visualize_history(
            history=history, 
            save_dir='visualizations', 
            model_name=model_name
        )
        
        # 4. Load Best Model & Test
        print(f"--- Testing {model_name} ---")
        weights_path = f"model_params/{model_name}_best_weights.pth"
        best_model = load_trained_model(
            model_class=ModelClass, 
            weights_path=weights_path, 
            num_classes=num_classes, 
            device=device
        )
        
        metrics = test_model(
            model=best_model, 
            model_name=model_name, 
            metrics_save_dir='model_params', 
            plots_save_dir='visualizations', 
            test_df=test_df, 
            img_dir='LD2_dataset/images', 
            device=device, 
            batch_size=32
        )
        
        print(f"Evaluation metrics for {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")

if __name__ == '__main__':
    main()