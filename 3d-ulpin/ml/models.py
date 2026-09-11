"""
PointNet++ / KPConv Models for 3D Cadastral Segmentation
- Building extraction from DSM/point cloud
- Floor segmentation from point cloud
- Vertical unit delineation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ============================================================================
# POINTNET++ BUILDING BLOCKS
# ============================================================================

class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (sampling + grouping + PointNet)"""
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # +3 for XYZ
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: (B, N, 3)
        points: (B, N, C) - features (can include XYZ)
        Returns: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape
        _, _, C_feat = points.shape
        
        if self.group_all:
            # Group all points
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            grouped_points = points.view(B, 1, N, C_feat)
        else:
            # Farthest point sampling
            fps_idx = self._farthest_point_sample(xyz, self.npoint)
            new_xyz = xyz[torch.arange(B).unsqueeze(-1), fps_idx]
            
            # Ball query
            grouped_xyz, grouped_points = self._ball_query(xyz, points, new_xyz)
        
        # Normalize grouped points
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        
        # Concat XYZ with features
        if points is not None:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        
        # PointNet on grouped points
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        
        for i, (conv, bn) in enumerate(zip(self.mlp_convs, self.mlp_bns)):
            grouped_points = F.relu(bn(conv(grouped_points)))
        
        # Max pooling
        new_points = torch.max(grouped_points, dim=2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        
        return new_xyz, new_points
    
    def _farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Farthest point sampling"""
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.full((B, N), float('inf'), device=xyz.device)
        farthest = torch.randint(0, N, (B,), device=xyz.device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, dim=-1)[1]
        
        return centroids
    
    def _ball_query(self, xyz: torch.Tensor, points: torch.Tensor, new_xyz: torch.Tensor):
        """Ball query for grouping"""
        B, N, _ = xyz.shape
        B, npoint, _ = new_xyz.shape
        
        grouped_xyz = torch.zeros(B, npoint, self.nsample, 3, device=xyz.device)
        grouped_points = torch.zeros(B, npoint, self.nsample, points.shape[-1], device=xyz.device)
        
        for b in range(B):
            for i in range(npoint):
                center = new_xyz[b, i]
                dist = torch.sum((xyz[b] - center) ** 2, dim=-1)
                idx = torch.topk(dist, self.nsample, largest=False)[1]
                grouped_xyz[b, i] = xyz[b, idx]
                grouped_points[b, i] = points[b, idx]
        
        return grouped_xyz, grouped_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation (upsampling)"""
    
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(
        self, 
        xyz1: torch.Tensor,  # (B, N1, 3) - target points
        xyz2: torch.Tensor,  # (B, N2, 3) - source points
        points1: torch.Tensor,  # (B, N1, C1)
        points2: torch.Tensor   # (B, N2, C2)
    ) -> torch.Tensor:
        """Interpolate features from xyz2 to xyz1"""
        B, N1, _ = xyz1.shape
        B, N2, _ = xyz2.shape
        
        if N2 == 1:
            # Global feature
            interpolated = points2.repeat(1, N1, 1)
        else:
            # Three nearest neighbor interpolation
            dist = torch.cdist(xyz1, xyz2)  # (B, N1, N2)
            dist, idx = torch.topk(dist, 3, dim=-1, largest=False)
            weight = 1.0 / (dist + 1e-8)
            weight = weight / weight.sum(dim=-1, keepdim=True)
            
            interpolated = torch.sum(
                weight.unsqueeze(-1) * points2[torch.arange(B).unsqueeze(-1).unsqueeze(-1), idx],
                dim=2
            )
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated
        
        new_points = new_points.permute(0, 2, 1)  # (B, C, N1)
        
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.permute(0, 2, 1)


# ============================================================================
# POINTNET++ FOR FLOOR SEGMENTATION
# ============================================================================

class PointNetPlusPlusFloorSeg(nn.Module):
    """
    PointNet++ for floor-level segmentation
    Input: (B, N, 6) - XYZ + RGB/Intensity
    Output: (B, N, num_classes) - per-point floor labels
    """
    
    def __init__(self, num_classes: int = 20, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channels, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024], group_all=True)
        
        # Decoder
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channels, [128, 128])
        
        # Classification head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        features: (B, N, C) - RGB, intensity, etc.
        Returns: (B, N, num_classes)
        """
        # Concatenate XYZ with features if needed
        if features is not None:
            points = torch.cat([xyz, features], dim=-1)
        else:
            points = xyz
        
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)
        
        # Classification
        x = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        
        return x.permute(0, 2, 1)


# ============================================================================
# KPConv FOR FLOOR SEGMENTATION (Alternative)
# ============================================================================

class KPConvLayer(nn.Module):
    """Kernel Point Convolution layer"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        radius: float = 0.1,
        dimension: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.dimension = dimension
        
        # Kernel points (learnable positions)
        self.kernel_points = nn.Parameter(torch.randn(kernel_size, dimension))
        
        # Weights
        self.weights = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Influence function (correlation)
        self.sigma = nn.Parameter(torch.ones(1) * radius / 3)
    
    def forward(
        self, 
        xyz: torch.Tensor,      # (N, 3)
        features: torch.Tensor, # (N, C_in)
        batch: torch.Tensor     # (N,) batch indices
    ) -> torch.Tensor:
        """
        Returns: (N, C_out)
        """
        N, C_in = features.shape
        device = xyz.device
        
        # For each point, find neighbors within radius
        # This is a simplified version - real KPConv uses custom CUDA ops
        out_features = torch.zeros(N, self.out_channels, device=device)
        
        # Compute pairwise distances within batch
        for b in batch.unique():
            mask = batch == b
            xyz_b = xyz[mask]
            feat_b = features[mask]
            n_b = xyz_b.shape[0]
            
            if n_b == 0:
                continue
            
            # Distance matrix
            dist = torch.cdist(xyz_b.unsqueeze(0), xyz_b.unsqueeze(0)).squeeze(0)
            
            # Neighbor mask
            neighbor_mask = (dist < self.radius) & (dist > 1e-6)
            
            for i in range(n_b):
                neighbors = torch.where(neighbor_mask[i])[0]
                if len(neighbors) == 0:
                    continue
                
                # Relative positions
                rel_pos = xyz_b[neighbors] - xyz_b[i:i+1]
                
                # Influence weights
                dist_neighbors = dist[i, neighbors]
                influence = torch.exp(-dist_neighbors ** 2 / (2 * self.sigma ** 2))
                
                # Kernel point distances
                kernel_dist = torch.cdist(
                    rel_pos.unsqueeze(0), 
                    self.kernel_points.unsqueeze(0)
                ).squeeze(0)
                
                # Kernel weights (Gaussian)
                kernel_weight = torch.exp(-kernel_dist ** 2 / (2 * self.sigma ** 2))
                
                # Weighted sum
                weighted_feat = feat_b[neighbors].unsqueeze(1) * \
                               kernel_weight.unsqueeze(-1) * \
                               influence.unsqueeze(-1).unsqueeze(-1)
                
                out_features[mask][i] = (weighted_feat * self.weights).sum(dim=(0, 1))
        
        return out_features + self.bias


class KPConvFloorSeg(nn.Module):
    """
    KPConv for floor segmentation
    More efficient for large point clouds than PointNet++
    """
    
    def __init__(self, num_classes: int = 20, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.conv1 = KPConvLayer(in_channels, 64, kernel_size=15, radius=0.1)
        self.conv2 = KPConvLayer(64, 128, kernel_size=15, radius=0.2)
        self.conv3 = KPConvLayer(128, 256, kernel_size=15, radius=0.4)
        self.conv4 = KPConvLayer(256, 512, kernel_size=15, radius=0.8)
        
        # Decoder (skip connections)
        self.deconv4 = KPConvLayer(512 + 256, 256, kernel_size=15, radius=0.8)
        self.deconv3 = KPConvLayer(256 + 128, 128, kernel_size=15, radius=0.4)
        self.deconv2 = KPConvLayer(128 + 64, 64, kernel_size=15, radius=0.2)
        self.deconv1 = KPConvLayer(64 + in_channels, 64, kernel_size=15, radius=0.1)
        
        # Classification
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
    
    def forward(
        self, 
        xyz: torch.Tensor,      # (N, 3)
        features: torch.Tensor, # (N, C)
        batch: torch.Tensor     # (N,)
    ) -> torch.Tensor:
        # Encoder
        f1 = F.relu(self.bn1(self.conv1(xyz, features, batch)))
        f2 = F.relu(self.bn2(self.conv2(xyz, f1, batch)))
        f3 = F.relu(self.bn3(self.conv3(xyz, f2, batch)))
        f4 = F.relu(self.bn4(self.conv4(xyz, f3, batch)))
        
        # Decoder with skip connections
        f4_up = F.relu(self.deconv4(xyz, torch.cat([f4, f3], dim=-1), batch))
        f3_up = F.relu(self.deconv3(xyz, torch.cat([f4_up, f2], dim=-1), batch))
        f2_up = F.relu(self.deconv2(xyz, torch.cat([f3_up, f1], dim=-1), batch))
        f1_up = F.relu(self.deconv1(xyz, torch.cat([f2_up, features], dim=-1), batch))
        
        # Classification
        return self.fc(f1_up)


# ============================================================================
# BUILDING EXTRACTION (2.5D from DSM/Ortho)
# ============================================================================

class BuildingExtractor(nn.Module):
    """
    U-Net style building footprint extraction from DSM + Ortho
    Input: (B, C, H, W) - DSM + RGB/Intensity
    Output: (B, 2, H, W) - Building mask + Boundary
    """
    
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, num_classes, 1)
    
    def _conv_block(self, in_c: int, out_c: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)


# ============================================================================
# MODEL FACTORY & EXPORT
# ============================================================================

def create_floor_segmenter(
    model_type: str = 'pointnet++',
    num_classes: int = 20,
    in_channels: int = 6
) -> nn.Module:
    """Create floor segmentation model"""
    if model_type == 'pointnet++':
        return PointNetPlusPlusFloorSeg(num_classes, in_channels)
    elif model_type == 'kpconv':
        return KPConvFloorSeg(num_classes, in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_building_extractor(in_channels: int = 4) -> nn.Module:
    """Create building footprint extractor"""
    return BuildingExtractor(in_channels=in_channels)


def export_to_onnx(model: nn.Module, input_shape: Tuple, output_path: str):
    """Export PyTorch model to ONNX"""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")


# Class names for floor segmentation
FLOOR_SEG_CLASSES = {
    0: 'ground',
    1: 'wall_exterior',
    2: 'floor_slab',
    3: 'roof',
    4: 'window',
    5: 'door',
    6: 'column',
    7: 'stair',
    8: 'interior_wall',
    9: 'ceiling',
    10: 'beam',
    11: 'railing',
    12: 'pipe',
    13: 'cable_tray',
    14: 'duct',
    15: 'equipment',
    16: 'furniture',
    17: 'vegetation',
    18: 'vehicle',
    19: 'clutter'
}


def get_class_weights(num_classes: int = 20) -> torch.Tensor:
    """Class weights for imbalanced cadastral data"""
    # Floors, walls, roofs are common; stairs, equipment rare
    weights = torch.ones(num_classes)
    weights[0] = 0.5    # ground (lots of points)
    weights[1] = 1.0    # wall
    weights[2] = 2.0    # floor_slab (important)
    weights[3] = 1.5    # roof
    weights[4] = 5.0    # window (rare)
    weights[5] = 10.0   # door (very rare)
    weights[6] = 8.0    # column
    weights[7] = 15.0   # stair
    weights[8] = 3.0    # interior_wall
    weights[9] = 4.0    # ceiling
    weights[10] = 6.0   # beam
    weights[11] = 8.0   # railing
    weights[12] = 10.0  # pipe
    weights[13] = 10.0  # cable_tray
    weights[14] = 10.0  # duct
    weights[15] = 12.0  # equipment
    weights[16] = 15.0  # furniture
    weights[17] = 2.0   # vegetation
    weights[18] = 10.0  # vehicle
    weights[19] = 5.0   # clutter
    return weights