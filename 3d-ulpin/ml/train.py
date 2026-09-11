"""
Training Pipeline for 3D Cadastral ML Models
- Synthetic data generation
- PointNet++/KPConv floor segmentation
- Building footprint extraction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import random

from ml.models import (
    PointNetPlusPlusFloorSeg, KPConvFloorSeg, BuildingExtractor,
    create_floor_segmenter, create_building_extractor,
    export_to_onnx, get_class_weights, FLOOR_SEG_CLASSES
)
from ml.synthetic_generator import SyntheticDataGenerator, SyntheticBuilding, generate_dataset


class CadastralPointCloudDataset(Dataset):
    """Dataset for floor segmentation from point clouds"""
    
    def __init__(
        self, 
        data_dir: str,
        split: str = 'train',
        num_points: int = 8192,
        augment: bool = True,
        class_mapping: Dict[int, int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        # Default: use all classes
        self.class_mapping = class_mapping or {i: i for i in range(20)}
        
        # Load file list
        self.files = list(self.data_dir.glob(f"{split}/*.npz"))
        if not self.files:
            # Try flat structure
            self.files = list(self.data_dir.glob("*.npz"))
        
        # Split if single directory
        if len(self.files) > 0 and split != 'all':
            random.shuffle(self.files)
            n = len(self.files)
            if split == 'train':
                self.files = self.files[:int(n * 0.8)]
            elif split == 'val':
                self.files = self.files[int(n * 0.8):int(n * 0.9)]
            elif split == 'test':
                self.files = self.files[int(n * 0.9):]
        
        print(f"{split}: {len(self.files)} files")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        points = data['points']      # (N, 6) XYZ + RGB
        labels = data['labels']      # (N,)
        
        # Remap labels
        new_labels = np.array([self.class_mapping.get(l, -1) for l in labels])
        
        # Filter valid labels
        valid = new_labels >= 0
        points = points[valid]
        new_labels = new_labels[valid]
        
        if len(points) == 0:
            # Return dummy
            return torch.zeros(self.num_points, 6), torch.full((self.num_points,), -1, dtype=torch.long)
        
        # Sample/ pad to num_points
        if len(points) > self.num_points:
            idxs = np.random.choice(len(points), self.num_points, replace=False)
            points = points[idxs]
            new_labels = new_labels[idxs]
        elif len(points) < self.num_points:
            # Pad with repetition
            repeat = self.num_points // len(points) + 1
            points = np.tile(points, (repeat, 1))[:self.num_points]
            new_labels = np.tile(new_labels, repeat)[:self.num_points]
        
        # Normalize XYZ (center at origin)
        xyz = points[:, :3].astype(np.float32)
        xyz = xyz - xyz.mean(axis=0)
        
        # Features (RGB/Intensity)
        features = points[:, 3:].astype(np.float32) / 255.0 if points[:, 3:].max() > 1 else points[:, 3:].astype(np.float32)
        
        # Augmentation
        if self.augment and self.split == 'train':
            xyz, features = self._augment(xyz, features)
        
        return (
            torch.from_numpy(np.concatenate([xyz, features], axis=1)),
            torch.from_numpy(new_labels.astype(np.int64))
        )
    
    def _augment(self, xyz: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Data augmentation"""
        # Random rotation around Z
        if random.random() < 0.5:
            angle = random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            xyz = xyz @ rot.T
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            xyz *= scale
        
        # Random jitter
        if random.random() < 0.5:
            xyz += np.random.normal(0, 0.01, xyz.shape)
        
        # Random dropout
        if random.random() < 0.3:
            mask = np.random.rand(len(xyz)) > 0.1
            xyz = xyz[mask]
            features = features[mask]
            # Resample to maintain count
            if len(xyz) < self.num_points:
                idxs = np.random.choice(len(xyz), self.num_points, replace=True)
                xyz = xyz[idxs]
                features = features[idxs]
        
        return xyz, features


class CadastralRasterDataset(Dataset):
    """Dataset for building extraction from DSM/Ortho"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        patch_size: int = 256,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # Expected structure: data_dir/split/images/*.tif, data_dir/split/masks/*.tif
        self.image_dir = self.data_dir / split / 'images'
        self.mask_dir = self.data_dir / split / 'masks'
        
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        print(f"{split}: {len(self.image_files)} image-mask pairs")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import rasterio
        
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)
        
        # Normalize image
        for c in range(image.shape[0]):
            p2, p98 = np.percentile(image[c], (2, 98))
            image[c] = np.clip((image[c] - p2) / (p98 - p2 + 1e-6), 0, 1)
        
        # Random crop
        H, W = image.shape[1:]
        if H > self.patch_size and W > self.patch_size:
            y = random.randint(0, H - self.patch_size)
            x = random.randint(0, W - self.patch_size)
            image = image[:, y:y+self.patch_size, x:x+self.patch_size]
            mask = mask[y:y+self.patch_size, x:x+self.patch_size]
        
        # Augmentation
        if self.augment and self.split == 'train':
            image, mask = self._augment(image, mask)
        
        return torch.from_numpy(image), torch.from_numpy(mask)
    
    def _augment(self, image: np.ndarray, mask: np.ndarray):
        # Flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Rotate 90 degrees
        if random.random() < 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
        return image, mask


def train_floor_segmentation(config: Dict):
    """Train floor segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Datasets
    train_dataset = CadastralPointCloudDataset(
        config['data_dir'],
        split='train',
        num_points=config['num_points'],
        augment=True
    )
    val_dataset = CadastralPointCloudDataset(
        config['data_dir'],
        split='val',
        num_points=config['num_points'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Model
    model = create_floor_segmenter(
        config['model_type'],
        num_classes=config['num_classes'],
        in_channels=config['in_channels']
    ).to(device)
    
    # Loss with class weights
    class_weights = get_class_weights(config['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # Training loop
    best_miou = 0.0
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_xyz, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            batch_xyz = batch_xyz.to(device)
            batch_labels = batch_labels.to(device)
            
            # Split XYZ and features
            xyz = batch_xyz[:, :, :3]
            features = batch_xyz[:, :, 3:]
            
            optimizer.zero_grad()
            
            if isinstance(model, PointNetPlusPlusFloorSeg):
                logits = model(xyz, features)
            else:  # KPConv
                batch_idx = torch.arange(batch_xyz.shape[0]).repeat_interleave(batch_xyz.shape[1]).to(device)
                logits = model(
                    xyz.view(-1, 3),
                    features.view(-1, features.shape[-1]),
                    batch_idx
                ).view(batch_xyz.shape[0], batch_xyz.shape[1], -1)
            
            loss = criterion(
                logits.permute(0, 2, 1),  # (B, C, N)
                batch_labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_xyz, batch_labels in val_loader:
                batch_xyz = batch_xyz.to(device)
                batch_labels = batch_labels.to(device)
                
                xyz = batch_xyz[:, :, :3]
                features = batch_xyz[:, :, 3:]
                
                if isinstance(model, PointNetPlusPlusFloorSeg):
                    logits = model(xyz, features)
                else:
                    batch_idx = torch.arange(batch_xyz.shape[0]).repeat_interleave(batch_xyz.shape[1]).to(device)
                    logits = model(
                        xyz.view(-1, 3),
                        features.view(-1, features.shape[-1]),
                        batch_idx
                    ).view(batch_xyz.shape[0], batch_xyz.shape[1], -1)
                
                loss = criterion(
                    logits.permute(0, 2, 1),
                    batch_labels
                )
                val_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
        
        # Compute mIoU
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        miou = compute_miou(all_preds, all_labels, config['num_classes'])
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}, mIoU={miou:.4f}")
        
        # Save best
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': miou,
                'config': config
            }, config['checkpoint_dir'] / 'best_model.pth')
            print(f"  -> Saved best model (mIoU={miou:.4f})")
    
    # Export to ONNX
    export_to_onnx(
        model,
        (1, config['num_points'], config['in_channels']),
        str(config['checkpoint_dir'] / 'floor_segmenter.onnx')
    )


def train_building_extraction(config: Dict):
    """Train building footprint extractor"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Datasets
    train_dataset = CadastralRasterDataset(
        config['data_dir'],
        split='train',
        patch_size=config['patch_size'],
        augment=True
    )
    val_dataset = CadastralRasterDataset(
        config['data_dir'],
        split='val',
        patch_size=config['patch_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Model
    model = create_building_extractor(config['in_channels']).to(device)
    
    # Loss (building + boundary)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]).to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_iou = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
        
        # Compute IoU for building class
        all_preds = torch.cat(all_preds).numpy()
        all_masks = torch.cat(all_masks).numpy()
        
        building_iou = compute_iou(all_preds, all_masks, 1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}, Building IoU={building_iou:.4f}")
        
        if building_iou > best_iou:
            best_iou = building_iou
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'building_iou': building_iou
            }, config['checkpoint_dir'] / 'building_extractor.pth')


def compute_miou(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Mean IoU ignoring class -1"""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum()
        union = (pred_c | label_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def compute_iou(preds: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    """IoU for specific class"""
    pred_c = preds == class_id
    label_c = labels == class_id
    intersection = (pred_c & label_c).sum()
    union = (pred_c | label_c).sum()
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['floor_seg', 'building_ext', 'generate_data'], default='floor_seg')
    parser.add_argument('--config', type=str, default='ml/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data/synthetic')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_type', choices=['pointnet++', 'kpconv'], default='pointnet++')
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--in_channels', type=int, default=6)
    parser.add_argument('--checkpoint_dir', type=str, default='ml/checkpoints')
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'model_type': args.model_type,
        'num_points': args.num_points,
        'num_classes': args.num_classes,
        'in_channels': args.in_channels,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'checkpoint_dir': Path(args.checkpoint_dir)
    }
    
    config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    
    if args.task == 'generate_data':
        generate_dataset(args.data_dir, n_buildings=1000, max_floors=20)
    elif args.task == 'floor_seg':
        train_floor_segmentation(config)
    elif args.task == 'building_ext':
        train_building_extraction(config)


if __name__ == '__main__':
    main()