"""
FER2013 Facial Expression Recognition - Training Script
Supports multiple architectures with WandB logging
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None, split='Training'):
        if 'Usage' in dataframe.columns:
            self.data = dataframe[dataframe['Usage'] == split].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = np.array(self.data.loc[idx, 'pixels'].split(), dtype=np.float32)
        image = pixels.reshape(48, 48)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 48, 48)
        image = image.repeat(3, 1, 1)  # (3, 48, 48) for pretrained models

        if self.transform:
            image = self.transform(image)

        label = int(self.data.loc[idx, 'emotion'])
        return image, label


def get_transforms(augment=True, img_size=48):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((img_size, img_size)),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        normalize,
    ])
    return train_transform, val_transform


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class TinyFERNet(nn.Module):
    """Architecture 1: Very small CNN - baseline (likely underfits)"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MediumFERNet(nn.Module):
    """Architecture 2: Medium CNN with BatchNorm and Dropout"""
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepFERNet(nn.Module):
    """Architecture 3: Deeper CNN with residual-style connections"""
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
            )

        self.block1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = conv_block(256, 256)
        self.pool4 = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return self.classifier(x)


class ResNet18FER(nn.Module):
    """Architecture 4: Transfer learning with ResNet18"""
    def __init__(self, num_classes=7, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


ARCHITECTURES = {
    'tiny':    TinyFERNet,
    'medium':  MediumFERNet,
    'deep':    DeepFERNet,
    'resnet18': ResNet18FER,
}


# ──────────────────────────────────────────────
# Diagnostics (forward / backward checks)
# ──────────────────────────────────────────────

def check_forward(model, device, img_size=48):
    """Sanity-check: one forward pass."""
    model.eval()
    dummy = torch.randn(4, 3, img_size, img_size).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"[Forward check] output shape: {out.shape}, min={out.min():.3f}, max={out.max():.3f}")
    assert out.shape == (4, 7), "Output shape mismatch!"
    print("[Forward check] PASSED ✓")
    return True


def check_backward(model, device, img_size=48):
    """Sanity-check: verify gradients flow."""
    model.train()
    dummy = torch.randn(4, 3, img_size, img_size).to(device)
    labels = torch.randint(0, 7, (4,)).to(device)
    criterion = nn.CrossEntropyLoss()
    out = model(dummy)
    loss = criterion(out, labels)
    loss.backward()
    # check at least one param has grad
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    print(f"[Backward check] loss={loss.item():.4f}, has_gradients={has_grad}")
    assert has_grad, "No gradients found!"
    print("[Backward check] PASSED ✓")
    model.zero_grad()
    return True


def overfit_single_batch(model, device, img_size=48, steps=100):
    """Can model overfit a tiny batch? (checks capacity)"""
    model.train()
    x = torch.randn(8, 3, img_size, img_size).to(device)
    y = torch.randint(0, 7, (8,)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"[Overfit check] final_loss={final_loss:.4f}, acc={acc:.2f}")
    model.zero_grad()
    return final_loss, acc


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc='Val', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_experiment(config):
    # ── WandB init ──
    run = wandb.init(
        project=config['wandb_project'],
        name=config['run_name'],
        config=config,
        tags=[config['architecture'], 'fer2013'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ──
    df = pd.read_csv(config['data_path'])
    img_size = config.get('img_size', 48)
    train_tf, val_tf = get_transforms(augment=config.get('augment', True), img_size=img_size)

    train_ds = FER2013Dataset(df, transform=train_tf, split='Training')
    val_ds   = FER2013Dataset(df, transform=val_tf,   split='PublicTest')

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ──
    arch_cls = ARCHITECTURES[config['architecture']]
    arch_kwargs = config.get('arch_kwargs', {})
    model = arch_cls(num_classes=7, **arch_kwargs).to(device)
    print(f"Model: {config['architecture']} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Sanity checks ──
    check_forward(model, device, img_size)
    check_backward(model, device, img_size)
    overfit_loss, overfit_acc = overfit_single_batch(model, device, img_size)
    wandb.log({'sanity/overfit_loss': overfit_loss, 'sanity/overfit_acc': overfit_acc})

    # ── Reinit model after overfit test ──
    model = arch_cls(num_classes=7, **arch_kwargs).to(device)

    # ── Optimizer & scheduler ──
    optimizer_name = config.get('optimizer', 'adam')
    lr = config['lr']
    wd = config.get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    scheduler_name = config.get('scheduler', 'cosine')
    epochs = config['epochs']

    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(
        weight=None if not config.get('class_weights') else
        torch.tensor([1.0, 5.0, 1.0, 0.5, 1.0, 1.0, 1.0]).to(device)
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Training loop ──
    best_val_acc = 0.
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 15)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, preds, labels_list = eval_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc':  train_acc,
            'val/loss':   val_loss,
            'val/acc':    val_acc,
            'train_val_loss_gap': val_loss - train_loss,
            'train_val_acc_gap':  train_acc - val_acc,
            'lr': current_lr,
        })

        print(f"Epoch {epoch:03d} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"LR={current_lr:.6f}")

        # Scheduler step
        if scheduler:
            if scheduler_name == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{config['run_name']}.pt")
            wandb.run.summary['best_val_acc'] = best_val_acc
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Final confusion matrix ──
    model.load_state_dict(torch.load(f"best_{config['run_name']}.pt"))
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)

    cm = confusion_matrix(final_labels, final_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=EMOTIONS.values(), yticklabels=EMOTIONS.values())
    ax.set_title(f'Confusion Matrix – {config["run_name"]} (acc={final_acc:.4f})')
    wandb.log({'confusion_matrix': wandb.Image(fig)})
    plt.close(fig)

    # Per-class accuracy
    report = classification_report(final_labels, final_preds,
                                   target_names=list(EMOTIONS.values()), output_dict=True)
    for cls_name, metrics in report.items():
        if isinstance(metrics, dict):
            wandb.log({f'class/{cls_name}/f1': metrics['f1-score'],
                       f'class/{cls_name}/precision': metrics['precision'],
                       f'class/{cls_name}/recall': metrics['recall']})

    wandb.run.summary['final_val_acc'] = final_acc
    wandb.finish()
    return best_val_acc


# ──────────────────────────────────────────────
# Experiment configs
# ──────────────────────────────────────────────

EXPERIMENTS = [
    # ── Exp 1: Tiny baseline (underfitting expected) ──
    {
        'run_name':      'exp01_tiny_baseline',
        'architecture':  'tiny',
        'epochs':        30,
        'batch_size':    64,
        'lr':            1e-3,
        'optimizer':     'adam',
        'scheduler':     'cosine',
        'augment':       False,
        'weight_decay':  0,
        'early_stop_patience': 10,
    },
    # ── Exp 2: Tiny + augmentation ──
    {
        'run_name':      'exp02_tiny_augmented',
        'architecture':  'tiny',
        'epochs':        30,
        'batch_size':    64,
        'lr':            1e-3,
        'optimizer':     'adam',
        'scheduler':     'cosine',
        'augment':       True,
        'weight_decay':  1e-4,
        'early_stop_patience': 10,
    },
    # ── Exp 3: Medium – no regularization (overfitting expected) ──
    {
        'run_name':      'exp03_medium_no_reg',
        'architecture':  'medium',
        'arch_kwargs':   {'dropout': 0.0},
        'epochs':        50,
        'batch_size':    64,
        'lr':            1e-3,
        'optimizer':     'adam',
        'scheduler':     'cosine',
        'augment':       False,
        'weight_decay':  0,
        'early_stop_patience': 20,
    },
    # ── Exp 4: Medium – regularized ──
    {
        'run_name':      'exp04_medium_regularized',
        'architecture':  'medium',
        'arch_kwargs':   {'dropout': 0.5},
        'epochs':        60,
        'batch_size':    64,
        'lr':            5e-4,
        'optimizer':     'adamw',
        'scheduler':     'cosine',
        'augment':       True,
        'weight_decay':  1e-3,
        'early_stop_patience': 15,
    },
    # ── Exp 5: Medium – SGD high LR ──
    {
        'run_name':      'exp05_medium_sgd',
        'architecture':  'medium',
        'arch_kwargs':   {'dropout': 0.4},
        'epochs':        60,
        'batch_size':    128,
        'lr':            0.01,
        'optimizer':     'sgd',
        'scheduler':     'step',
        'augment':       True,
        'weight_decay':  1e-4,
        'early_stop_patience': 15,
    },
    # ── Exp 6: Deep ──
    {
        'run_name':      'exp06_deep_cnn',
        'architecture':  'deep',
        'arch_kwargs':   {'dropout': 0.4},
        'epochs':        70,
        'batch_size':    64,
        'lr':            3e-4,
        'optimizer':     'adamw',
        'scheduler':     'cosine',
        'augment':       True,
        'weight_decay':  1e-4,
        'early_stop_patience': 20,
    },
    # ── Exp 7: Deep – class weighted loss ──
    {
        'run_name':      'exp07_deep_class_weights',
        'architecture':  'deep',
        'arch_kwargs':   {'dropout': 0.4},
        'epochs':        70,
        'batch_size':    64,
        'lr':            3e-4,
        'optimizer':     'adamw',
        'scheduler':     'plateau',
        'augment':       True,
        'weight_decay':  1e-4,
        'class_weights': True,
        'early_stop_patience': 20,
    },
    # ── Exp 8: ResNet18 frozen backbone ──
    {
        'run_name':      'exp08_resnet18_frozen',
        'architecture':  'resnet18',
        'arch_kwargs':   {'freeze_backbone': True},
        'img_size':      64,
        'epochs':        30,
        'batch_size':    64,
        'lr':            1e-3,
        'optimizer':     'adam',
        'scheduler':     'cosine',
        'augment':       True,
        'weight_decay':  1e-4,
        'early_stop_patience': 10,
    },
    # ── Exp 9: ResNet18 fine-tuned ──
    {
        'run_name':      'exp09_resnet18_finetune',
        'architecture':  'resnet18',
        'arch_kwargs':   {'freeze_backbone': False},
        'img_size':      64,
        'epochs':        50,
        'batch_size':    32,
        'lr':            1e-4,
        'optimizer':     'adamw',
        'scheduler':     'cosine',
        'augment':       True,
        'weight_decay':  1e-4,
        'early_stop_patience': 15,
    },
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='train.csv')
    parser.add_argument('--wandb_project', default='fer2013-experiments')
    parser.add_argument('--exp_idx', type=int, default=None,
                        help='Run single experiment by index (0-based). Default: run all.')
    args = parser.parse_args()

    exps = [EXPERIMENTS[args.exp_idx]] if args.exp_idx is not None else EXPERIMENTS

    for cfg in exps:
        cfg['data_path'] = args.data_path
        cfg['wandb_project'] = args.wandb_project
        print(f"\n{'='*60}")
        print(f"  Running: {cfg['run_name']}")
        print(f"{'='*60}")
        try:
            best_acc = run_experiment(cfg)
            print(f"  ✓ Best val acc: {best_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            raise
