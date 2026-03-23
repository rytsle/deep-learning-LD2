import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FolderDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples  # [(path, label), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_all_samples(data_dir: str):
    class_names = sorted(os.listdir(data_dir))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]
    label_map = {name: idx for idx, name in enumerate(class_names)}

    paths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_map[class_name])

    return paths, labels


def create_dataloaders(data_dir, fraction, train_transform, val_transform, test_size=0.2, batch_size=32, random_state=42):
    paths, labels = get_all_samples(data_dir)
    class_names = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))])

    if fraction < 1.0:
        paths, _, labels, _ = train_test_split(
            paths, labels,
            train_size=fraction,
            stratify=labels,
            random_state=random_state
        )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))

    train_dataset = FolderDataset(train_samples, transform=train_transform)
    val_dataset = FolderDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names
