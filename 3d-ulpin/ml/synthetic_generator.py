"""
Synthetic 3D Cadastral Data Generator
Generates training data for ML models from CityGML-like parametric buildings
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


@dataclass
class SyntheticBuilding:
    """Parametric building definition"""
    building_id: str
    footprint: np.ndarray  # [N, 2] polygon vertices
    num_floors: int
    floor_height: float
    ground_z: float
    roof_type: str  # flat|gabled|hipped|pyramid
    units_per_floor: int
    unit_layout: str  # grid|perimeter|corridor


class SyntheticDataGenerator:
    """Generate synthetic 3D point clouds with labels for training"""
    
    def __init__(
        self,
        point_density: float = 50.0,  # points per m²
        noise_std: float = 0.02,      # Gaussian noise (m)
        outlier_ratio: float = 0.01   # random outliers
    ):
        self.point_density = point_density
        self.noise_std = noise_std
        self.outlier_ratio = outlier_ratio
    
    def generate_building(
        self,
        building: SyntheticBuilding,
        include_interior: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate labeled point cloud for a building
        Returns: (points[N, 6], labels[N], metadata)
        Labels: 0=ground, 1=wall, 2=floor_slab, 3=roof, 4=window, 5=door, 6=column, 7=stair, 8=interior_wall
        """
        all_points = []
        all_labels = []
        
        # Generate each floor
        for floor_idx in range(building.num_floors):
            z_base = building.ground_z + floor_idx * building.floor_height
            z_top = z_base + building.floor_height
            
            # Floor slab
            slab_pts, slab_labels = self._generate_floor_slab(
                building.footprint, z_base, z_top, floor_idx
            )
            all_points.append(slab_pts)
            all_labels.append(slab_labels)
            
            # Walls
            wall_pts, wall_labels = self._generate_walls(
                building.footprint, z_base, z_top, floor_idx, building
            )
            all_points.append(wall_pts)
            all_labels.append(wall_labels)
            
            # Interior walls (if requested)
            if include_interior:
                int_pts, int_labels = self._generate_interior_walls(
                    building.footprint, z_base, z_top, floor_idx, building
                )
                all_points.append(int_pts)
                all_labels.append(int_labels)
        
        # Roof
        roof_pts, roof_labels = self._generate_roof(
            building.footprint, 
            building.ground_z + building.num_floors * building.floor_height,
            building
        )
        all_points.append(roof_pts)
        all_labels.append(roof_labels)
        
        # Ground around building
        ground_pts, ground_labels = self._generate_ground(building.footprint, building.ground_z)
        all_points.append(ground_pts)
        all_labels.append(ground_labels)
        
        # Combine
        points = np.vstack(all_points)
        labels = np.hstack(all_labels)
        
        # Add noise
        points[:, :3] += np.random.normal(0, self.noise_std, points[:, :3].shape)
        
        # Add outliers
        n_outliers = int(len(points) * self.outlier_ratio)
        if n_outliers > 0:
            outlier_pts = np.random.uniform(
                points[:, :3].min(axis=0) - 10,
                points[:, :3].max(axis=0) + 10,
                (n_outliers, 3)
            )
            outlier_feats = np.random.rand(n_outliers, 3)  # RGB/intensity
            outlier_labels = np.full(n_outliers, -1)  # Ignore class
            points = np.vstack([points, np.hstack([outlier_pts, outlier_feats])])
            labels = np.hstack([labels, outlier_labels])
        
        # Shuffle
        idx = np.random.permutation(len(points))
        points = points[idx]
        labels = labels[idx]
        
        metadata = {
            'building_id': building.building_id,
            'num_floors': building.num_floors,
            'floor_height': building.floor_height,
            'classes_present': np.unique(labels[labels >= 0]).tolist(),
            'total_points': len(points)
        }
        
        return points, labels, metadata
    
    def _generate_floor_slab(
        self, 
        footprint: np.ndarray, 
        z_base: float, 
        z_top: float,
        floor_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate floor slab points (top and bottom surfaces)"""
        # Triangulate footprint for sampling
        from scipy.spatial import Delaunay
        
        # Add small buffer for slab thickness
        slab_thickness = 0.2
        z_bottom = z_base
        z_top_slab = z_base + slab_thickness
        
        # Sample points in polygon
        n_points = int(self._polygon_area(footprint) * self.point_density * 2)  # top + bottom
        pts_xy = self._sample_in_polygon(footprint, n_points)
        
        # Bottom surface
        bottom_pts = np.column_stack([
            pts_xy, 
            np.full(len(pts_xy), z_bottom),
            np.random.rand(len(pts_xy), 3) * 0.5 + 0.3  # RGB
        ])
        
        # Top surface
        top_pts = np.column_stack([
            pts_xy,
            np.full(len(pts_xy), z_top_slab),
            np.random.rand(len(pts_xy), 3) * 0.5 + 0.3
        ])
        
        points = np.vstack([bottom_pts, top_pts])
        labels = np.hstack([
            np.full(len(bottom_pts), 2),  # floor_slab
            np.full(len(top_pts), 2)
        ])
        
        return points, labels
    
    def _generate_walls(
        self,
        footprint: np.ndarray,
        z_base: float,
        z_top: float,
        floor_idx: int,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate exterior wall points with windows/doors"""
        all_pts = []
        all_labels = []
        
        n_vertices = len(footprint)
        perimeter = self._polygon_perimeter(footprint)
        n_points = int(perimeter * building.floor_height * self.point_density)
        
        for i in range(n_vertices):
            v1 = footprint[i]
            v2 = footprint[(i + 1) % n_vertices]
            edge_len = np.linalg.norm(v2 - v1)
            
            if edge_len < 0.5:
                continue
            
            edge_points = int(n_points * edge_len / perimeter)
            
            # Sample along edge
            t = np.random.rand(edge_points)
            xy = v1 + t[:, None] * (v2 - v1)
            
            # Sample height
            z = np.random.rand(edge_points) * (z_top - z_base) + z_base
            
            # Determine if window/door (ground floor has doors, all have windows)
            is_opening = np.random.rand(edge_points) < 0.15
            is_door = (floor_idx == 0) & (np.random.rand(edge_points) < 0.05)
            
            pts = np.column_stack([
                xy,
                z,
                np.random.rand(edge_points, 3) * 0.4 + 0.4
            ])
            
            labels = np.where(
                is_door, 5,
                np.where(is_opening, 4, 1)  # wall
            )
            
            all_pts.append(pts)
            all_labels.append(labels)
        
        # Columns at corners
        for v in footprint:
            n_col = max(5, int(building.floor_height * self.point_density / 10))
            z_col = np.linspace(z_base, z_top, n_col)
            xy_col = np.tile(v, (n_col, 1))
            col_pts = np.column_stack([
                xy_col,
                z_col,
                np.random.rand(n_col, 3) * 0.3 + 0.5
            ])
            all_pts.append(col_pts)
            all_labels.append(np.full(n_col, 6))  # column
        
        if all_pts:
            return np.vstack(all_pts), np.hstack(all_labels)
        return np.empty((0, 6)), np.empty(0, dtype=int)
    
    def _generate_interior_walls(
        self,
        footprint: np.ndarray,
        z_base: float,
        z_top: float,
        floor_idx: int,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate interior partition walls"""
        # Simple grid-based interior walls
        minx, miny = footprint.min(axis=0)
        maxx, maxy = footprint.max(axis=0)
        width = maxx - minx
        height = maxy - miny
        
        all_pts = []
        all_labels = []
        
        # Grid lines
        n_divisions = max(2, int(np.sqrt(building.units_per_floor)))
        
        # Vertical grid lines
        for i in range(1, n_divisions):
            x = minx + i * width / n_divisions
            y_vals = np.linspace(miny, maxy, int(height * self.point_density))
            x_vals = np.full_like(y_vals, x)
            
            # Check if inside footprint
            from matplotlib.path import Path
            path = Path(footprint)
            inside = path.contains_points(np.column_stack([x_vals, y_vals]))
            
            if inside.any():
                xy = np.column_stack([x_vals[inside], y_vals[inside]])
                z = np.random.rand(len(xy)) * (z_top - z_base) + z_base
                pts = np.column_stack([
                    xy, z, np.random.rand(len(xy), 3) * 0.3 + 0.6
                ])
                all_pts.append(pts)
                all_labels.append(np.full(len(pts), 8))  # interior_wall
        
        # Horizontal grid lines
        for i in range(1, n_divisions):
            y = miny + i * height / n_divisions
            x_vals = np.linspace(minx, maxx, int(width * self.point_density))
            y_vals = np.full_like(x_vals, y)
            
            from matplotlib.path import Path
            path = Path(footprint)
            inside = path.contains_points(np.column_stack([x_vals, y_vals]))
            
            if inside.any():
                xy = np.column_stack([x_vals[inside], y_vals[inside]])
                z = np.random.rand(len(xy)) * (z_top - z_base) + z_base
                pts = np.column_stack([
                    xy, z, np.random.rand(len(xy), 3) * 0.3 + 0.6
                ])
                all_pts.append(pts)
                all_labels.append(np.full(len(pts), 8))
        
        if all_pts:
            return np.vstack(all_pts), np.hstack(all_labels)
        return np.empty((0, 6)), np.empty(0, dtype=int)
    
    def _generate_roof(
        self,
        footprint: np.ndarray,
        roof_base_z: float,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate roof points based on roof type"""
        if building.roof_type == 'flat':
            return self._generate_flat_roof(footprint, roof_base_z)
        elif building.roof_type == 'gabled':
            return self._generate_gabled_roof(footprint, roof_base_z, building)
        elif building.roof_type == 'hipped':
            return self._generate_hipped_roof(footprint, roof_base_z, building)
        else:
            return self._generate_pyramid_roof(footprint, roof_base_z, building)
    
    def _generate_flat_roof(self, footprint: np.ndarray, z: float) -> Tuple[np.ndarray, np.ndarray]:
        n_points = int(self._polygon_area(footprint) * self.point_density)
        pts_xy = self._sample_in_polygon(footprint, n_points)
        pts = np.column_stack([
            pts_xy,
            np.full(len(pts_xy), z),
            np.random.rand(len(pts_xy), 3) * 0.3 + 0.4
        ])
        return pts, np.full(len(pts), 3)  # roof
    
    def _generate_gabled_roof(
        self, 
        footprint: np.ndarray, 
        z_base: float,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Simplified: two sloped planes meeting at ridge
        minx, miny = footprint.min(axis=0)
        maxx, maxy = footprint.max(axis=0)
        ridge_y = (miny + maxy) / 2
        roof_height = max(2.0, building.floor_height * 0.8)
        
        all_pts = []
        all_labels = []
        
        # Sample points on roof
        n_points = int(self._polygon_area(footprint) * self.point_density * 1.5)
        pts_xy = self._sample_in_polygon(footprint, n_points)
        
        # Compute Z based on distance from ridge
        for xy in pts_xy:
            y = xy[1]
            dist_from_ridge = abs(y - ridge_y)
            max_dist = (maxy - miny) / 2
            z = z_base + roof_height * (1 - dist_from_ridge / max_dist)
            z = max(z, z_base)
            
            all_pts.append([xy[0], xy[1], z, *np.random.rand(3) * 0.3 + 0.4])
        
        points = np.array(all_pts)
        labels = np.full(len(points), 3)
        
        # Add ridge line
        ridge_pts = []
        for x in np.linspace(minx, maxx, 20):
            ridge_pts.append([x, ridge_y, z_base + roof_height, *np.random.rand(3) * 0.2 + 0.7])
        if ridge_pts:
            ridge_pts = np.array(ridge_pts)
            points = np.vstack([points, ridge_pts])
            labels = np.hstack([labels, np.full(len(ridge_pts), 3)])
        
        return points, labels
    
    def _generate_hipped_roof(
        self, 
        footprint: np.ndarray, 
        z_base: float,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Four sloped planes
        minx, miny = footprint.min(axis=0)
        maxx, maxy = footprint.max(axis=0)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        roof_height = max(2.0, building.floor_height * 0.8)
        
        n_points = int(self._polygon_area(footprint) * self.point_density * 1.5)
        pts_xy = self._sample_in_polygon(footprint, n_points)
        
        all_pts = []
        for xy in pts_xy:
            dx = abs(xy[0] - cx) / ((maxx - minx) / 2)
            dy = abs(xy[1] - cy) / ((maxy - miny) / 2)
            d = max(dx, dy)
            z = z_base + roof_height * (1 - d)
            z = max(z, z_base)
            all_pts.append([xy[0], xy[1], z, *np.random.rand(3) * 0.3 + 0.4])
        
        points = np.array(all_pts)
        return points, np.full(len(points), 3)
    
    def _generate_pyramid_roof(
        self, 
        footprint: np.ndarray, 
        z_base: float,
        building: SyntheticBuilding
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Pyramid roof
        minx, miny = footprint.min(axis=0)
        maxx, maxy = footprint.max(axis=0)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        roof_height = max(2.5, building.floor_height)
        
        n_points = int(self._polygon_area(footprint) * self.point_density * 1.5)
        pts_xy = self._sample_in_polygon(footprint, n_points)
        
        all_pts = []
        for xy in pts_xy:
            dx = abs(xy[0] - cx) / ((maxx - minx) / 2)
            dy = abs(xy[1] - cy) / ((maxy - miny) / 2)
            d = max(dx, dy)
            z = z_base + roof_height * (1 - d)
            z = max(z, z_base)
            all_pts.append([xy[0], xy[1], z, *np.random.rand(3) * 0.3 + 0.4])
        
        points = np.array(all_pts)
        return points, np.full(len(points), 3)
    
    def _generate_ground(
        self, 
        footprint: np.ndarray, 
        ground_z: float,
        buffer: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ground points around building"""
        minx, miny = footprint.min(axis=0) - buffer
        maxx, maxy = footprint.max(axis=0) + buffer
        
        area = (maxx - minx) * (maxy - miny)
        building_area = self._polygon_area(footprint)
        ground_area = area - building_area
        
        n_points = int(ground_area * self.point_density * 0.5)  # Lower density
        
        # Sample in bounding box, reject if in building
        from matplotlib.path import Path
        path = Path(footprint)
        
        pts = []
        while len(pts) < n_points:
            batch = 10000
            xy = np.random.uniform([minx, miny], [maxx, maxy], (batch, 2))
            outside = ~path.contains_points(xy)
            valid = xy[outside]
            pts.extend(valid)
            if len(pts) >= n_points:
                break
        
        pts_xy = np.array(pts[:n_points])
        z = np.full(len(pts_xy), ground_z) + np.random.normal(0, 0.01, len(pts_xy))
        
        points = np.column_stack([
            pts_xy, z, np.random.rand(len(pts_xy), 3) * 0.4 + 0.2
        ])
        return points, np.full(len(points), 0)  # ground
    
    def _polygon_area(self, poly: np.ndarray) -> float:
        """Shoelace formula"""
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def _polygon_perimeter(self, poly: np.ndarray) -> float:
        diffs = np.diff(poly, axis=0, append=poly[:1])
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def _sample_in_polygon(self, poly: np.ndarray, n: int) -> np.ndarray:
        """Uniform sampling in polygon using triangulation"""
        from scipy.spatial import Delaunay
        from matplotlib.path import Path
        
        path = Path(poly)
        minx, miny = poly.min(axis=0)
        maxx, maxy = poly.max(axis=0)
        
        pts = []
        while len(pts) < n:
            batch = min(n * 3, 10000)
            xy = np.random.uniform([minx, miny], [maxx, maxy], (batch, 2))
            inside = path.contains_points(xy)
            pts.extend(xy[inside])
        
        return np.array(pts[:n])


def generate_dataset(
    output_dir: str,
    n_buildings: int = 1000,
    max_floors: int = 20,
    point_density: float = 50.0
):
    """Generate full synthetic dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDataGenerator(point_density=point_density)
    
    # Building types
    roof_types = ['flat', 'gabled', 'hipped', 'pyramid']
    layouts = ['grid', 'perimeter', 'corridor']
    
    dataset_info = {
        'classes': {
            0: 'ground',
            1: 'wall',
            2: 'floor_slab',
            3: 'roof',
            4: 'window',
            5: 'door',
            6: 'column',
            7: 'stair',
            8: 'interior_wall'
        },
        'buildings': []
    }
    
    for i in range(n_buildings):
        # Random building params
        n_floors = random.randint(1, max_floors)
        floor_h = random.uniform(2.8, 3.5)
        ground_z = random.uniform(0, 500)
        roof_type = random.choice(roof_types)
        layout = random.choice(layouts)
        units = random.randint(2, 12)
        
        # Random footprint (rectangle with noise)
        w = random.uniform(15, 60)
        h = random.uniform(15, 60)
        cx, cy = random.uniform(-500, 500), random.uniform(-500, 500)
        
        # Rectangle with slight noise
        footprint = np.array([
            [cx - w/2, cy - h/2],
            [cx + w/2, cy - h/2],
            [cx + w/2, cy + h/2],
            [cx - w/2, cy + h/2]
        ])
        
        # Add noise to vertices
        footprint += np.random.normal(0, 0.5, footprint.shape)
        
        building = SyntheticBuilding(
            building_id=f"SYNTH_{i:06d}",
            footprint=footprint,
            num_floors=n_floors,
            floor_height=floor_h,
            ground_z=ground_z,
            roof_type=roof_type,
            units_per_floor=units,
            unit_layout=layout
        )
        
        points, labels, meta = generator.generate_building(building, include_interior=True)
        
        # Save as NPZ
        np.savez_compressed(
            output_path / f"{building.building_id}.npz",
            points=points.astype(np.float32),
            labels=labels.astype(np.int32)
        )
        
        dataset_info['buildings'].append(meta)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_buildings} buildings")
    
    # Save dataset info
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset generated at {output_path}")
    return dataset_info


if __name__ == "__main__":
    generate_dataset("data/synthetic", n_buildings=100, max_floors=10)