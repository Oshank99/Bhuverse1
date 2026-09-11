"""
PDAL Pipelines for LiDAR/Point Cloud Processing
- Ground classification (SMRF) -> DTM/DSM
- Building extraction
- Noise removal
"""
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ProcessingResult:
    success: bool
    output_files: Dict[str, str]
    metrics: Dict
    logs: str
    error: Optional[str] = None


class PDALPipeline:
    """PDAL pipeline executor with common workflows"""
    
    def __init__(self, pdal_path: str = "pdal"):
        self.pdal_path = pdal_path
        self.work_dir = Path(tempfile.gettempdir()) / "pdal_ulpin"
        self.work_dir.mkdir(exist_ok=True)
    
    def run_pipeline(self, pipeline: dict, output_log: bool = True) -> ProcessingResult:
        """Execute a PDAL pipeline JSON"""
        pipeline_file = self.work_dir / f"pipeline_{os.urandom(4).hex()}.json"
        
        try:
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline, f, indent=2)
            
            cmd = [self.pdal_path, "pipeline", str(pipeline_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            logs = result.stdout + "\n" + result.stderr
            
            if result.returncode != 0:
                return ProcessingResult(
                    success=False,
                    output_files={},
                    metrics={},
                    logs=logs,
                    error=f"PDAL failed with code {result.returncode}"
                )
            
            # Extract metrics from logs
            metrics = self._parse_metrics(logs)
            
            return ProcessingResult(
                success=True,
                output_files={},
                metrics=metrics,
                logs=logs
            )
        except subprocess.TimeoutExpired:
            return ProcessingResult(
                success=False,
                output_files={},
                metrics={},
                logs="",
                error="Pipeline timeout"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                output_files={},
                metrics={},
                logs="",
                error=str(e)
            )
        finally:
            if pipeline_file.exists():
                pipeline_file.unlink()
    
    def _parse_metrics(self, logs: str) -> Dict:
        """Parse PDAL output for metrics"""
        metrics = {}
        for line in logs.split('\n'):
            if 'points' in line.lower() and ('read' in line.lower() or 'write' in line.lower()):
                # Try to extract point counts
                pass
        return metrics


# ============================================================================
# PREDEFINED PIPELINES
# ============================================================================

def create_ground_classification_pipeline(
    input_las: str,
    output_las: str,
    dtm_tif: str = None,
    dsm_tif: str = None,
    smrf_scalar: float = 1.2,
    smrf_slope: float = 0.2,
    smrf_threshold: float = 0.45,
    smrf_window: int = 16
) -> dict:
    """SMRF ground classification -> DTM/DSM"""
    pipeline = {
        "pipeline": [
            input_las,
            {
                "type": "filters.smrf",
                "scalar": smrf_scalar,
                "slope": smrf_slope,
                "threshold": smrf_threshold,
                "window": smrf_window
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Keep only ground
            }
        ]
    }
    
    # Add DTM writer if requested
    if dtm_tif:
        pipeline["pipeline"].append({
            "type": "writers.gdal",
            "filename": dtm_tif,
            "output_type": "idw",
            "resolution": 1.0,
            "window_size": 3
        })
    
    # Add classified LAS writer
    pipeline["pipeline"].append({
        "type": "writers.las",
        "filename": output_las,
        "compression": "laszip"
    })
    
    # Add DSM from first returns (separate pipeline)
    return pipeline


def create_dsm_pipeline(
    input_las: str,
    dsm_tif: str,
    resolution: float = 1.0
) -> dict:
    """Generate DSM from first returns"""
    return {
        "pipeline": [
            input_las,
            {
                "type": "filters.range",
                "limits": "ReturnNumber[1:1]"  # First returns only
            },
            {
                "type": "writers.gdal",
                "filename": dsm_tif,
                "output_type": "max",
                "resolution": resolution,
                "window_size": 3
            }
        ]
    }


def create_building_extraction_pipeline(
    input_las: str,
    output_las: str,
    building_footprints: str = None,
    min_height: float = 3.0,
    min_area: float = 20.0
) -> dict:
    """Extract building points using height above ground + clustering"""
    pipeline = {
        "pipeline": [
            input_las,
            # Compute height above ground (requires ground classification first)
            {
                "type": "filters.hag_nn"
            },
            # Filter by height
            {
                "type": "filters.range",
                "limits": f"HeightAboveGround[{min_height}:]"
            },
            # Cluster buildings (Euclidean clustering)
            {
                "type": "filters.cluster",
                "min_points": 20,
                "distance": 2.0
            }
        ]
    }
    
    if building_footprints:
        pipeline["pipeline"].append({
            "type": "writers.gpkg",
            "filename": building_footprints,
            "layer_name": "buildings",
            "geometry_type": "POLYGON"
        })
    
    pipeline["pipeline"].append({
        "type": "writers.las",
        "filename": output_las,
        "compression": "laszip"
    })
    
    return pipeline


def create_noise_removal_pipeline(
    input_las: str,
    output_las: str,
    method: str = "statistical",
    mean_k: int = 8,
    multiplier: float = 2.0
) -> dict:
    """Remove noise/outliers from point cloud"""
    pipeline = {
        "pipeline": [
            input_las,
        ]
    }
    
    if method == "statistical":
        pipeline["pipeline"].append({
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": mean_k,
            "multiplier": multiplier
        })
    elif method == "radius":
        pipeline["pipeline"].append({
            "type": "filters.outlier",
            "method": "radius",
            "radius": 1.0,
            "min_k": 5
        })
    
    pipeline["pipeline"].append({
        "type": "writers.las",
        "filename": output_las,
        "compression": "laszip"
    })
    
    return pipeline


def create_reprojection_pipeline(
    input_las: str,
    output_las: str,
    target_crs: str = "EPSG:4326"
) -> dict:
    """Reproject point cloud to target CRS"""
    return {
        "pipeline": [
            input_las,
            {
                "type": "filters.reprojection",
                "out_srs": target_crs
            },
            {
                "type": "writers.las",
                "filename": output_las,
                "compression": "laszip"
            }
        ]
    }


def create_chm_pipeline(
    input_las: str,
    chm_tif: str,
    resolution: float = 0.5
) -> dict:
    """Canopy Height Model (DSM - DTM)"""
    return {
        "pipeline": [
            input_las,
            {
                "type": "filters.hag_nn"
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[0.5:]"  # Vegetation threshold
            },
            {
                "type": "writers.gdal",
                "filename": chm_tif,
                "output_type": "max",
                "resolution": resolution
            }
        ]
    }


def create_tile_pipeline(
    input_las: str,
    output_dir: str,
    tile_size: float = 1000.0,
    buffer: float = 50.0
) -> dict:
    """Tile large point cloud into smaller chunks"""
    return {
        "pipeline": [
            input_las,
            {
                "type": "filters.tile",
                "length": tile_size,
                "buffer": buffer
            },
            {
                "type": "writers.las",
                "filename": f"{output_dir}/tile_#.las",
                "compression": "laszip"
            }
        ]
    }


def create_full_processing_pipeline(
    input_las: str,
    output_dir: str,
    target_crs: str = "EPSG:4326"
) -> List[dict]:
    """Complete processing chain: reproject -> denoise -> ground -> DTM/DSM -> buildings -> CHM"""
    pipelines = []
    
    base_name = Path(input_las).stem
    
    # Stage 1: Reproject
    reprojected = f"{output_dir}/{base_name}_reprojected.laz"
    pipelines.append(create_reprojection_pipeline(input_las, reprojected, target_crs))
    
    # Stage 2: Denoise
    denoised = f"{output_dir}/{base_name}_denoised.laz"
    pipelines.append(create_noise_removal_pipeline(reprojected, denoised))
    
    # Stage 3: Ground classification + DTM
    ground_classified = f"{output_dir}/{base_name}_ground.laz"
    dtm = f"{output_dir}/{base_name}_dtm.tif"
    pipelines.append(create_ground_classification_pipeline(denoised, ground_classified, dtm_tif=dtm))
    
    # Stage 4: DSM
    dsm = f"{output_dir}/{base_name}_dsm.tif"
    pipelines.append(create_dsm_pipeline(denoised, dsm))
    
    # Stage 5: Building extraction
    buildings_las = f"{output_dir}/{base_name}_buildings.laz"
    buildings_gpkg = f"{output_dir}/{base_name}_buildings.gpkg"
    pipelines.append(create_building_extraction_pipeline(denoised, buildings_las, buildings_gpkg))
    
    # Stage 6: CHM
    chm = f"{output_dir}/{base_name}_chm.tif"
    pipelines.append(create_chm_pipeline(denoised, chm))
    
    return pipelines


# Python-based processing (for complex operations)
class PointCloudProcessor:
    """Python-based point cloud processing using PDAL Python bindings + numpy"""
    
    def __init__(self):
        try:
            import pdal
            self.pdal = pdal
        except ImportError:
            self.pdal = None
    
    def execute_pipeline(self, pipeline_json: dict) -> np.ndarray:
        """Execute pipeline and return numpy array"""
        if self.pdal is None:
            raise RuntimeError("PDAL Python bindings not available")
        
        pipeline = self.pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        return pipeline.arrays[0]
    
    def compute_building_footprints(self, points: np.ndarray, epsg: int = 4326) -> List[dict]:
        """Extract building footprints from classified building points"""
        from sklearn.cluster import DBSCAN
        from shapely.geometry import MultiPoint, Polygon
        from shapely.ops import unary_union
        
        # Filter building points (Classification == 6 for buildings)
        building_pts = points[points['Classification'] == 6]
        
        if len(building_pts) < 10:
            return []
        
        # XY coordinates for clustering
        coords = np.column_stack([building_pts['X'], building_pts['Y']])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=2.0, min_samples=20).fit(coords)
        labels = clustering.labels_
        
        footprints = []
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_pts = building_pts[labels == label]
            xy = np.column_stack([cluster_pts['X'], cluster_pts['Y']])
            
            # Convex hull / alpha shape
            multipoint = MultiPoint(xy)
            hull = multipoint.convex_hull
            
            if hull.area > 20:  # Minimum area
                footprints.append({
                    'geometry': hull,
                    'point_count': len(cluster_pts),
                    'avg_height': np.mean(cluster_pts['Z']),
                    'min_z': np.min(cluster_pts['Z']),
                    'max_z': np.max(cluster_pts['Z'])
                })
        
        return footprints
    
    def segment_floors(self, points: np.ndarray, floor_height: float = 3.0) -> np.ndarray:
        """Segment point cloud into floors based on Z"""
        z = points['Z']
        ground_z = np.percentile(z, 5)  # Estimate ground level
        
        # Assign floor numbers
        floor_numbers = np.floor((z - ground_z) / floor_height).astype(int)
        floor_numbers[floor_numbers < 0] = -1  # Basement
        
        # Add as new dimension
        points['FloorLevel'] = floor_numbers
        return points