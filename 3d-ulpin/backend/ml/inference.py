"""
ONNX Model Serving for 3D Cadastral Inference
- Floor segmentation
- Building extraction
- Vertical delineation
"""
import onnxruntime as ort
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class InferenceResult:
    """Model inference result"""
    success: bool
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    metadata: Dict = None
    error: Optional[str] = None


class FloorSegmenterONNX:
    """ONNX Runtime inference for floor segmentation"""
    
    def __init__(
        self,
        model_path: str,
        providers: List[str] = None,
        num_classes: int = 20
    ):
        self.num_classes = num_classes
        
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.expected_points = input_shape[1] if len(input_shape) > 1 else 8192
        self.expected_channels = input_shape[2] if len(input_shape) > 2 else 6
        
        print(f"Loaded floor segmenter: {model_path}")
        print(f"  Expected input: {input_shape}")
        print(f"  Providers: {self.session.get_providers()}")
    
    def predict(
        self, 
        points: np.ndarray,  # (N, 6) XYZ + RGB
        batch_size: int = 1
    ) -> InferenceResult:
        """
        Run floor segmentation on point cloud
        Handles point clouds larger than expected_points by tiling
        """
        try:
            N, C = points.shape
            
            if C != self.expected_channels:
                return InferenceResult(
                    success=False,
                    error=f"Expected {self.expected_channels} channels, got {C}"
                )
            
            if N <= self.expected_points:
                # Single batch
                return self._predict_batch(points[None, ...])
            else:
                # Tile and merge
                return self._predict_tiled(points)
                
        except Exception as e:
            return InferenceResult(success=False, error=str(e))
    
    def _predict_batch(self, batch: np.ndarray) -> InferenceResult:
        """Predict on single batch (1, N, C)"""
        # Normalize XYZ
        xyz = batch[:, :, :3].copy()
        xyz = xyz - xyz.mean(axis=1, keepdims=True)
        
        # Normalize features (assuming 0-255 or 0-1)
        features = batch[:, :, 3:].copy()
        if features.max() > 1.0:
            features = features / 255.0
        
        input_data = np.concatenate([xyz, features], axis=-1).astype(np.float32)
        
        # Run inference
        logits = self.session.run([self.output_name], {self.input_name: input_data})[0]
        
        # (1, N, num_classes) -> probabilities
        probs = self._softmax(logits[0])
        preds = np.argmax(probs, axis=-1)
        
        return InferenceResult(
            success=True,
            predictions=preds,
            probabilities=probs,
            metadata={'n_points': len(preds)}
        )
    
    def _predict_tiled(self, points: np.ndarray) -> InferenceResult:
        """Predict on large point cloud by tiling"""
        N = len(points)
        tile_size = self.expected_points
        stride = tile_size // 2  # 50% overlap
        
        all_preds = np.full(N, -1, dtype=np.int32)
        all_probs = np.zeros((N, self.num_classes), dtype=np.float32)
        vote_counts = np.zeros(N, dtype=np.int32)
        
        for start in range(0, N, stride):
            end = min(start + tile_size, N)
            if end - start < tile_size // 4:  # Skip tiny tiles
                continue
            
            tile = points[start:end]
            # Pad if needed
            if len(tile) < tile_size:
                pad = tile_size - len(tile)
                tile = np.vstack([tile, tile[:pad]])
            
            result = self._predict_batch(tile[None, ...])
            if not result.success:
                continue
            
            tile_len = end - start
            all_preds[start:end] = result.predictions[:tile_len]
            all_probs[start:end] += result.probabilities[:tile_len]
            vote_counts[start:end] += 1
        
        # Average probabilities where multiple votes
        mask = vote_counts > 0
        all_probs[mask] /= vote_counts[mask, None]
        
        # Final prediction from averaged probs
        final_preds = np.argmax(all_probs, axis=-1)
        final_preds[~mask] = -1
        
        return InferenceResult(
            success=True,
            predictions=final_preds,
            probabilities=all_probs,
            metadata={'n_points': N, 'tiles_processed': (N + stride - 1) // stride}
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)


class BuildingExtractorONNX:
    """ONNX Runtime inference for building footprint extraction"""
    
    def __init__(
        self,
        model_path: str,
        providers: List[str] = None
    ):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_channels = input_shape[1] if len(input_shape) > 1 else 4
        
        print(f"Loaded building extractor: {model_path}")
        print(f"  Expected input: {input_shape}")
    
    def predict(
        self,
        image: np.ndarray,  # (C, H, W) or (H, W, C)
        patch_size: int = 256,
        overlap: int = 32
    ) -> InferenceResult:
        """Extract building footprints from DSM/Ortho"""
        try:
            # Ensure CHW format
            if image.ndim == 3 and image.shape[-1] == self.input_channels:
                image = np.transpose(image, (2, 0, 1))
            elif image.ndim == 3 and image.shape[0] != self.input_channels:
                return InferenceResult(
                    success=False,
                    error=f"Expected {self.input_channels} channels, got {image.shape[0]}"
                )
            
            C, H, W = image.shape
            
            if H <= patch_size and W <= patch_size:
                return self._predict_patch(image)
            else:
                return self._predict_tiled(image, patch_size, overlap)
                
        except Exception as e:
            return InferenceResult(success=False, error=str(e))
    
    def _predict_patch(self, image: np.ndarray) -> InferenceResult:
        """Predict on single patch"""
        # Normalize
        img_norm = image.astype(np.float32)
        for c in range(img_norm.shape[0]):
            p2, p98 = np.percentile(img_norm[c], (2, 98))
            img_norm[c] = np.clip((img_norm[c] - p2) / (p98 - p2 + 1e-6), 0, 1)
        
        input_data = img_norm[None, ...]  # (1, C, H, W)
        
        logits = self.session.run([self.output_name], {self.input_name: input_data})[0]
        probs = self._softmax(logits[0])
        preds = np.argmax(probs, axis=0)
        
        return InferenceResult(
            success=True,
            predictions=preds,
            probabilities=probs,
            metadata={'shape': preds.shape}
        )
    
    def _predict_tiled(self, image: np.ndarray, patch_size: int, overlap: int) -> InferenceResult:
        """Tiled prediction for large images"""
        C, H, W = image.shape
        stride = patch_size - overlap
        
        full_probs = np.zeros((2, H, W), dtype=np.float32)  # 2 classes: bg, building
        vote_counts = np.zeros((H, W), dtype=np.int32)
        
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                
                if y_end - y < patch_size // 4 or x_end - x < patch_size // 4:
                    continue
                
                patch = image[:, y:y_end, x:x_end]
                
                # Pad if needed
                ph, pw = patch.shape[1], patch.shape[2]
                if ph < patch_size or pw < patch_size:
                    pad_h = patch_size - ph
                    pad_w = patch_size - pw
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                
                result = self._predict_patch(patch)
                if not result.success:
                    continue
                
                # Crop back to valid region
                tile_probs = result.probabilities[:, :ph, :pw]
                
                full_probs[:, y:y_end, x:x_end] += tile_probs
                vote_counts[y:y_end, x:x_end] += 1
        
        # Average
        mask = vote_counts > 0
        full_probs[:, mask] /= vote_counts[mask]
        
        final_preds = np.argmax(full_probs, axis=0)
        
        return InferenceResult(
            success=True,
            predictions=final_preds,
            probabilities=full_probs,
            metadata={'shape': final_preds.shape}
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0, keepdims=True)


class VerticalDelineatorML:
    """
    ML-enhanced vertical delineation
    Uses floor segmentation results to refine 3D units
    """
    
    def __init__(
        self,
        floor_segmenter: FloorSegmenterONNX,
        building_extractor: BuildingExtractorONNX = None
    ):
        self.floor_segmenter = floor_segmenter
        self.building_extractor = building_extractor
    
    def delineate_from_pointcloud(
        self,
        points: np.ndarray,  # (N, 6) XYZ + RGB
        building_footprint: np.ndarray = None  # (M, 2) polygon
    ) -> Dict:
        """
        Full pipeline: point cloud -> floor segmentation -> 3D units
        """
        # 1. Floor segmentation
        seg_result = self.floor_segmenter.predict(points)
        if not seg_result.success:
            return {'success': False, 'error': seg_result.error}
        
        labels = seg_result.predictions
        
        # 2. Extract floor slabs (class 2)
        floor_mask = labels == 2
        floor_points = points[floor_mask]
        
        if len(floor_points) < 100:
            return {'success': False, 'error': 'Insufficient floor points detected'}
        
        # 3. Cluster floor points by Z to get floor levels
        floor_levels = self._cluster_floors(floor_points)
        
        # 4. Generate 3D units for each floor
        units = []
        for level_idx, (z_base, z_top, level_points) in enumerate(floor_levels):
            # Project to 2D and get unit polygons
            unit_polygons = self._extract_units_2d(level_points)
            
            for unit_poly in unit_polygons:
                units.append({
                    'floor_level': level_idx,
                    'z_base': z_base,
                    'z_top': z_top,
                    'polygon': unit_poly,
                    'area': self._polygon_area(unit_poly)
                })
        
        return {
            'success': True,
            'floor_levels': floor_levels,
            'units': units,
            'segmentation': {
                'labels': labels.tolist(),
                'probabilities': seg_result.probabilities.tolist() if seg_result.probabilities is not None else None
            }
        }
    
    def _cluster_floors(self, floor_points: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        """Cluster floor points by Z coordinate"""
        from sklearn.cluster import DBSCAN
        
        z = floor_points[:, 2]
        
        # Use DBSCAN on Z only
        clustering = DBSCAN(eps=0.5, min_samples=50).fit(z.reshape(-1, 1))
        labels = clustering.labels_
        
        levels = []
        for label in np.unique(labels):
            if label == -1:
                continue
            
            level_pts = floor_points[labels == label]
            z_min, z_max = level_pts[:, 2].min(), level_pts[:, 2].max()
            z_base = (z_min + z_max) / 2 - 0.1  # Slab center - half thickness
            z_top = z_base + 0.2  # Slab thickness
            
            levels.append((z_base, z_top, level_pts))
        
        # Sort by Z
        levels.sort(key=lambda x: x[0])
        return levels
    
    def _extract_units_2d(self, level_points: np.ndarray) -> List[np.ndarray]:
        """Extract unit polygons from floor points using clustering"""
        from sklearn.cluster import DBSCAN
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        
        xy = level_points[:, :2]
        
        # Cluster into units
        clustering = DBSCAN(eps=3.0, min_samples=30).fit(xy)
        labels = clustering.labels_
        
        polygons = []
        for label in np.unique(labels):
            if label == -1:
                continue
            
            unit_pts = xy[labels == label]
            if len(unit_pts) < 10:
                continue
            
            try:
                hull = ConvexHull(unit_pts)
                poly = Polygon(unit_pts[hull.vertices])
                if poly.area > 5.0:  # Min 5 m²
                    polygons.append(np.array(poly.exterior.coords))
            except:
                pass
        
        return polygons
    
    def _polygon_area(self, poly: np.ndarray) -> float:
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_models(
    floor_segmenter_path: str = "ml/checkpoints/floor_segmenter.onnx",
    building_extractor_path: str = "ml/checkpoints/building_extractor.onnx"
) -> Tuple[FloorSegmenterONNX, BuildingExtractorONNX]:
    """Load all models for inference"""
    fs = FloorSegmenterONNX(floor_segmenter_path) if Path(floor_segmenter_path).exists() else None
    be = BuildingExtractorONNX(building_extractor_path) if Path(building_extractor_path).exists() else None
    return fs, be


# Integration with backend ingestion pipeline
async def run_ml_inference(
    source_type: str,
    file_path: str,
    models: Tuple[FloorSegmenterONNX, BuildingExtractorONNX]
) -> Dict:
    """Run ML inference as part of ingestion pipeline"""
    fs, be = models
    
    if source_type == 'lidar':
        import laspy
        las = laspy.read(file_path)
        points = np.column_stack([
            las.x, las.y, las.z,
            las.red if hasattr(las, 'red') else np.zeros(len(las.x)),
            las.green if hasattr(las, 'green') else np.zeros(len(las.x)),
            las.blue if hasattr(las, 'blue') else np.zeros(len(las.x))
        ])
        
        # Normalize colors
        if points[:, 3:].max() > 255:
            points[:, 3:] = points[:, 3:] / 65535.0 * 255
        
        result = fs.predict(points)
        if result.success:
            # Process results for vertical delineation
            delineator = VerticalDelineatorML(fs, be)
            return delineator.delineate_from_pointcloud(points)
    
    return {'success': False, 'error': f'Unsupported source type: {source_type}'}