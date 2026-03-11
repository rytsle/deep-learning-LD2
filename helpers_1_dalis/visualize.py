import os
import matplotlib.pyplot as plt

def visualize_history(history, save_dir='/visualizations/train', model_name='model'):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = history.get('best_epoch', None)

    def mark_best(ax, best_epoch, y_values):
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.6, label=f'Best epoch ({best_epoch})')
            ax.plot(best_epoch, y_values[best_epoch - 1], 'r*', markersize=12, zorder=5)

    # 1. Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Train Loss')
    ax.plot(epochs, history['val_loss'], label='Val Loss')
    mark_best(ax, best_epoch, history['val_loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_loss.png'), dpi=150)
    plt.close(fig)

    # 2. Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax.plot(epochs, history['val_acc'], label='Val Accuracy')
    mark_best(ax, best_epoch, history['val_acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'), dpi=150)
    plt.close(fig)

    # 3. F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_f1'], label='Train F1')
    ax.plot(epochs, history['val_f1'], label='Val F1')
    mark_best(ax, best_epoch, history['val_f1'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 (macro)')
    ax.set_title('Train vs Validation F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_f1.png'), dpi=150)
    plt.close(fig)

    # 4. Val F1 and Val Accuracy together
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['val_f1'], label='Val F1')
    ax.plot(epochs, history['val_acc'], label='Val Accuracy')
    mark_best(ax, best_epoch, history['val_f1'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation F1 vs Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_val_f1_vs_acc.png'), dpi=150)
    plt.close(fig)