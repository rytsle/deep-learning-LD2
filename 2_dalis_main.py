from torchvision import transforms
from helpers_2_dalis.dataloader import create_dataloaders
from helpers_2_dalis.train_evaluate import get_device, build_resnet18, train_model_resnet, test_model_resnet
from helpers_2_dalis.visualize import plot_experiment_results

DATA_PERCENTAGES = [10, 25, 50, 75, 100]
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
DATA_DIR = 'custom_data'
WEIGHTS_DIR = 'model_params_2dalis'
METRICS_DIR = 'model_params_2dalis'
VIZ_DIR = 'visualizations_2dalis'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

aug_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VARIANTS = [
    ('scratch_noaug',    False, base_transform),
    ('scratch_aug',      False, aug_transform),
    ('pretrained_noaug', True,  base_transform),
    ('pretrained_aug',   True,  aug_transform),
]

device = get_device()
print(f"Using device: {device}")

all_results = {}

for pct in DATA_PERCENTAGES:
    fraction = pct / 100.0
    all_results[pct] = {}

    for variant_key, use_pretrained, train_transform in VARIANTS:
        model_name = f'resnet18_{variant_key}_pct{pct}'
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        train_loader, val_loader, class_names = create_dataloaders(
            data_dir=DATA_DIR,
            fraction=fraction,
            train_transform=train_transform,
            val_transform=base_transform,
            batch_size=BATCH_SIZE,
        )

        model = build_resnet18(pretrained=use_pretrained, num_classes=4)

        model, history = train_model_resnet(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=EPOCHS,
            lr=LR,
            device=device,
            save_weights=(not use_pretrained),
            weights_save_dir=WEIGHTS_DIR,
        )

        test_model_resnet(
            model=model,
            model_name=model_name,
            test_loader=val_loader,
            device=device,
            metrics_save_dir=METRICS_DIR,
        )

        all_results[pct][variant_key] = history

plot_experiment_results(all_results, save_dir=VIZ_DIR, filename='resnet18_experiment_results.png')
print("\nDone!")
