import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, ViTForImageClassification

import numpy as np
from collections import Counter
import random
from torchvision import transforms
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from pathlib import Path
from tqdm.auto import tqdm

# --- Config ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = Path("artifacts")
# Corrected DATA_DIR to match the 'via-cervix/Data/...' structure
DATA_DIR = ARTIFACTS_DIR / "via-cervix" / "Data"
RESULTS_DIR = ARTIFACTS_DIR / "training_runs"

CLASS_NAMES = ["Negative", "Positive", "Suspicious cancer"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

MODEL_NAME = "google/vit-base-patch16-224"
NUM_LABELS = len(CLASS_NAMES)
BASE_MODEL_PATH = ARTIFACTS_DIR / "base_model"
TRAINED_MODEL_PATH = RESULTS_DIR / "best_model.pth"

SEED = 42
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CANCER_PENALTY = 15.0
NUM_SPLITS = 5
BATCH_SIZE = 16

FILE_ID = "1lFvuTpzdfSAckyjtHZzE2HWqsH25sa1q"
ZIP_PATH = ARTIFACTS_DIR / "via-cervix.zip"
EXTRACT_DIR = ARTIFACTS_DIR / "via-cervix"

# --- Loss Functions ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def create_cost_matrix(n_classes, cancer_penalty=15.0):
    matrix = torch.ones(n_classes, n_classes)
    matrix[CLASS_TO_IDX["Suspicious cancer"], :] = cancer_penalty
    matrix.fill_diagonal_(1)
    return matrix

class CostSensitiveLoss(nn.Module):
    def __init__(self, n_classes, cancer_penalty=15.0):
        super().__init__()
        self.cost_matrix = create_cost_matrix(n_classes, cancer_penalty).to(DEVICE)

    def forward(self, logits, targets):
        softmax = F.softmax(logits, dim=1)
        costs = self.cost_matrix[targets]
        return (softmax * costs).sum(1).mean()

# --- Data Augmentation and Dataset ---

class AggressiveAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

    def __call__(self, img):
        return self.transform(img)

class ImbalancedDataset(Dataset):
    def __init__(self, samples, val=False):
        self.samples = samples
        self.val = val
        self.augmentor = AggressiveAugmentation()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if not self.val:
            img = self.augmentor(img)
        return self.transform(img), label

# --- Trainer Class ---

class MultiStageTrainer:
    def __init__(self, model, device, class_names):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.cancer_idx = class_names.index("Suspicious cancer")
        self.focal_loss = FocalLoss()
        self.cost_sensitive_loss = CostSensitiveLoss(len(class_names), CANCER_PENALTY)

    def stage1_binary_training(self, train_loader, epochs=3):
        print("\n--- Stage 1: Binary Classification (Cancer vs. Non-Cancer) ---")
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Stage 1, Epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                binary_labels = (labels == self.cancer_idx).long()
                
                optimizer.zero_grad()
                logits = self.model(images).logits
                loss = F.cross_entropy(logits, binary_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Stage 1, Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    def stage2_multiclass_training(self, train_loader, val_loader, epochs=10):
        print("\n--- Stage 2: Multiclass Classification ---")
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
        best_recall = 0.0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Stage 2, Epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images).logits
                loss = self.focal_loss(logits, labels) + self.cost_sensitive_loss(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            val_recall = self.evaluate(val_loader)
            print(f"Stage 2, Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}, Cancer Recall: {val_recall:.3f}")
            if val_recall > best_recall:
                best_recall = val_recall
                best_model_state = self.model.state_dict()
        
        return best_recall, best_model_state

    def evaluate(self, val_loader):
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images).logits
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        return report[str(self.cancer_idx)]['recall']

# --- Helper and Main Functions ---

def list_images_by_class(root_dir, class_names):
    samples = []
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = root_dir / class_name
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                samples.append((img_path, class_to_idx[class_name]))
    return samples

def main():
    """Main training function."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = list_images_by_class(DATA_DIR, CLASS_NAMES)
    
    if not all_samples:
        print(f"Error: No images found in {DATA_DIR}. Please check the path, class names, and directory structure.")
        return

    labels = np.array([s[1] for s in all_samples])

    skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
    best_overall_recall = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        print(f"\n{'='*20} Fold {fold+1}/{NUM_SPLITS} {'='*20}")

        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        train_dataset = ImbalancedDataset(train_samples)
        val_dataset = ImbalancedDataset(val_samples, val=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
        trainer = MultiStageTrainer(model, DEVICE, CLASS_NAMES)

        trainer.stage1_binary_training(train_loader, epochs=3)
        best_fold_recall, best_fold_model_state = trainer.stage2_multiclass_training(
            train_loader, val_loader, epochs=EPOCHS
        )

        if best_fold_recall > best_overall_recall:
            best_overall_recall = best_fold_recall
            best_model_state = best_fold_model_state
            model.load_state_dict(best_fold_model_state)
            save_evaluation_files(model, val_loader, DEVICE)

    if best_model_state:
        torch.save(best_model_state, TRAINED_MODEL_PATH)
        print(f"\nBest model saved to {TRAINED_MODEL_PATH} with cancer recall: {best_overall_recall:.3f}")

def save_evaluation_files(model, val_loader, device):
    print("Saving evaluation files for the best fold...")
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images).logits
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    np.save(RESULTS_DIR / "eval_labels.npy", np.array(all_labels))
    np.save(RESULTS_DIR / "eval_probs.npy", np.array(all_probs))
    print("Evaluation files saved successfully.")

if __name__ == "__main__":
    main()