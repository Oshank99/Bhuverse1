"""
Vertical Parcel Delineation - Extrude 2D floor polygons to 3D volumes
Rule-based + point cloud fusion approach
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union, transform
from shapely.affinity import translate
import math


@dataclass
class VolumetricUnit:
    """3D volumetric parcel unit"""
    suid: str
    uln: str
    label: str
    unit_type: str  # parcel|building|floor|unit|utility|airspace|subsurface
    geometry_wkt: str  # POLYHEDRALSURFACE Z
    z_min: float
    z_max: float
    floor_level: int
    parent_suid: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VerticalDelineator:
    """
    Convert 2D floor polygons + heights to 3D volumetric units
    Supports:
    - Simple extrusion (floor polygons * height)
    - Point cloud fusion (refine boundaries with LiDAR)
    - Complex roof forms
    - Underground structures
    """
    
    def __init__(
        self,
        default_floor_height: float = 3.0,
        ground_z: float = 0.0,
        tolerance: float = 0.01
    ):
        self.default_floor_height = default_floor_height
        self.ground_z = ground_z
        self.tolerance = tolerance
    
    def extrude_floor_plan(
        self,
        building_id: str,
        floor_polygons: List[Dict],  # [{'level', 'polygon_wkt', 'unit_id', 'unit_type', 'height', 'z_base'}, ...]
        ulpin_generator
    ) -> List[VolumetricUnit]:
        """
        Extrude floor polygons to 3D volumes
        """
        units = []
        
        # Sort by level
        floor_polygons.sort(key=lambda f: f.get('z_base', f.get('level', 0) * self.default_floor_height))
        
        # Create parcel volume (from ground to roof)
        parcel_poly = self._merge_floor_footprints(floor_polygons)
        if not parcel_poly:
            return units
        
        # Generate parcel ULPIN from actual centroid
        centroid = parcel_poly.centroid
        parcel_ulpin = ulpin_generator.generate_parcel_ulpin(centroid.x, centroid.y)
        
        parcel_unit = self._create_parcel_volume(
            building_id, parcel_poly, parcel_ulpin, floor_polygons
        )
        units.append(parcel_unit)
        
        # Create building volume
        building_ulpin = ulpin_generator.generate_building_ulpin(parcel_ulpin, 1)
        building_unit = self._create_building_volume(
            building_id, parcel_poly, building_ulpin, parcel_unit.suid, floor_polygons
        )
        units.append(building_unit)
        
        # Create floor volumes
        for i, fp in enumerate(floor_polygons):
            level = fp.get('level', i)
            z_base = fp.get('z_base', level * self.default_floor_height)
            height = fp.get('height', self.default_floor_height)
            
            floor_ulpin = ulpin_generator.generate_floor_ulpin(building_ulpin, level)
            
            floor_unit = self._create_floor_volume(
                building_id, fp, floor_ulpin, building_unit.suid, level, z_base, height
            )
            units.append(floor_unit)
            
            # Create unit volumes within floor
            unit_polygons = self._split_floor_into_units(fp)
            for j, unit_poly in enumerate(unit_polygons):
                unit_ulpin = ulpin_generator.generate_unit_ulpin(
                    floor_ulpin, j + 1,
                    x=unit_poly.centroid.x,
                    y=unit_poly.centroid.y,
                    z=z_base + height / 2
                )
                
                unit_volume = self._create_unit_volume(
                    building_id, unit_poly, unit_ulpin, floor_unit.suid,
                    level, z_base, height, fp.get('unit_type', 'residential')
                )
                units.append(unit_volume)
        
        return units
    
    def _merge_floor_footprints(self, floor_polygons: List[Dict]) -> Optional[Polygon]:
        """Merge all floor polygons to get building footprint"""
        from shapely.wkt import loads
        
        polys = []
        for fp in floor_polygons:
            try:
                poly = loads(fp['polygon_wkt'])
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
            except:
                pass
        
        if not polys:
            return None
        
        union = unary_union(polys)
        if union.geom_type == 'Polygon':
            return union
        elif union.geom_type == 'MultiPolygon':
            # Return largest polygon
            return max(union.geoms, key=lambda p: p.area)
        return None
    
    def _split_floor_into_units(self, floor_polygon: Dict) -> List[Polygon]:
        """Split floor polygon into individual units based on building type and layout"""
        from shapely.wkt import loads
        from shapely.geometry import box
        
        poly = loads(floor_polygon['polygon_wkt'])
        unit_type = floor_polygon.get('unit_type', 'residential')
        layout = floor_polygon.get('unit_layout', 'grid')
        units_per_floor = floor_polygon.get('units_per_floor')
        
        # If polygon has holes, each ring could be a unit
        if poly.geom_type == 'Polygon':
            # Get area in square meters
            area_sqm = self._area_sqm(poly)
            
            # Estimate units per floor if not provided
            if units_per_floor is None:
                units_per_floor = self._estimate_units_per_floor(area_sqm, unit_type)
            
            # Split based on layout
            if layout == 'grid':
                return self._split_grid(poly, units_per_floor)
            elif layout == 'perimeter':
                return self._split_perimeter(poly, units_per_floor)
            elif layout == 'corridor':
                return self._split_corridor(poly, units_per_floor)
            else:
                return [poly]
        elif poly.geom_type == 'MultiPolygon':
            return list(poly.geoms)
        return [poly]
    
    def _area_sqm(self, polygon: Polygon) -> float:
        """Calculate polygon area in square meters (approximate at centroid latitude)"""
        centroid = polygon.centroid
        lat = centroid.y
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(lat)
        deg_lat_m = 111000.0
        deg_lon_m = 111000.0 * math.cos(math.radians(lat))
        return polygon.area * deg_lat_m * deg_lon_m
    
    def _estimate_units_per_floor(self, area_sqm: float, unit_type: str) -> int:
        """Estimate number of units based on floor area in sqm and unit type"""
        if unit_type == 'residential':
            # ~80-120 m² per flat
            return max(1, round(area_sqm / 100))
        elif unit_type == 'commercial':
            # Larger units, ~200-500 m²
            return max(1, round(area_sqm / 300))
        elif unit_type == 'mixed':
            return max(2, round(area_sqm / 150))
        elif unit_type == 'apartments':
            # High density, ~60-90 m² per flat
            return max(1, round(area_sqm / 75))
        else:
            return max(1, round(area_sqm / 100))
    
    def _split_grid(self, polygon: Polygon, num_units: int) -> List[Polygon]:
        """Split polygon into grid layout"""
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        
        # Calculate grid dimensions
        cols = max(1, int(round(num_units ** 0.5)))
        rows = max(1, (num_units + cols - 1) // cols)
        
        cell_width = width / cols
        cell_height = height / rows
        
        units = []
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= num_units:
                    break
                cell = box(
                    minx + j * cell_width,
                    miny + i * cell_height,
                    minx + (j + 1) * cell_width,
                    miny + (i + 1) * cell_height
                )
                intersection = polygon.intersection(cell)
                if self._area_sqm(intersection) > 1.0:  # Min 1 m²
                    units.append(intersection)
                count += 1
        return units
    
    def _split_perimeter(self, polygon: Polygon, num_units: int) -> List[Polygon]:
        """Split polygon into perimeter units around central core"""
        centroid = polygon.centroid
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds
        
        # Create wedge segments from centroid
        units = []
        for i in range(num_units):
            angle_start = i * 2 * math.pi / num_units
            angle_end = (i + 1) * 2 * math.pi / num_units
            
            # Create wedge polygon
            wedge_coords = [centroid]
            
            # Add points along boundary
            exterior = polygon.exterior
            coords = list(exterior.coords)
            if len(coords) < 3:
                return [polygon]
            
            # Split by angle
            for x, y in coords:
                dx, dy = x - centroid.x, y - centroid.y
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                if angle_start <= angle < angle_end:
                    wedge_coords.append((x, y))
            
            if len(wedge_coords) >= 4:  # At least triangle + centroid
                try:
                    wedge = Polygon(wedge_coords)
                    intersection = polygon.intersection(wedge)
                    if self._area_sqm(intersection) > 1.0:
                        units.append(intersection)
                except:
                    pass
        
        if not units:
            return [polygon]
        return units
    
    def _split_corridor(self, polygon: Polygon, num_units: int) -> List[Polygon]:
        """Split polygon into corridor layout (units on both sides of central corridor)"""
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        
        # Determine corridor orientation (along longer dimension)
        if width > height:
            # Horizontal corridor
            corridor_width = min(2.0, width * 0.15)  # 1.5-2m corridor
            unit_depth = (width - corridor_width) / 2
            units_per_side = max(1, num_units // 2)
            
            units = []
            for side in [0, 1]:  # 0 = left, 1 = right
                for i in range(units_per_side):
                    y_start = miny + i * height / units_per_side
                    y_end = miny + (i + 1) * height / units_per_side
                    
                    if side == 0:
                        cell = box(minx, y_start, minx + unit_depth, y_end)
                    else:
                        cell = box(maxx - unit_depth, y_start, maxx, y_end)
                    
                    intersection = polygon.intersection(cell)
                    if self._area_sqm(intersection) > 1.0:
                        units.append(intersection)
        else:
            # Vertical corridor
            corridor_width = min(2.0, height * 0.15)
            unit_depth = (height - corridor_width) / 2
            units_per_side = max(1, num_units // 2)
            
            units = []
            for side in [0, 1]:
                for i in range(units_per_side):
                    x_start = minx + i * width / units_per_side
                    x_end = minx + (i + 1) * width / units_per_side
                    
                    if side == 0:
                        cell = box(x_start, miny, x_end, miny + unit_depth)
                    else:
                        cell = box(x_start, maxy - unit_depth, x_end, maxy)
                    
                    intersection = polygon.intersection(cell)
                    if self._area_sqm(intersection) > 1.0:
                        units.append(intersection)
        
        if not units:
            return [polygon]
        return units
    
    def _create_parcel_volume(
        self,
        building_id: str,
        footprint: Polygon,
        ulpin: str,
        floor_polygons: List[Dict]
    ) -> VolumetricUnit:
        """Create parcel volume (ground to roof)"""
        z_max = max(
            fp.get('z_base', 0) + fp.get('height', self.default_floor_height)
            for fp in floor_polygons
        )
        
        geometry = self._extrude_polygon(footprint, self.ground_z, z_max)
        
        return VolumetricUnit(
            suid=f"parcel-{building_id}",
            uln=ulpin,
            label=f"Parcel {building_id}",
            unit_type="parcel",
            geometry_wkt=geometry,
            z_min=self.ground_z,
            z_max=z_max,
            floor_level=0,
            metadata={'building_id': building_id}
        )
    
    def _create_building_volume(
        self,
        building_id: str,
        footprint: Polygon,
        ulpin: str,
        parent_suid: str,
        floor_polygons: List[Dict]
    ) -> VolumetricUnit:
        """Create building volume"""
        z_max = max(
            fp.get('z_base', 0) + fp.get('height', self.default_floor_height)
            for fp in floor_polygons
        )
        
        geometry = self._extrude_polygon(footprint, self.ground_z, z_max)
        
        return VolumetricUnit(
            suid=f"building-{building_id}",
            uln=ulpin,
            label=f"Building {building_id}",
            unit_type="building",
            geometry_wkt=geometry,
            z_min=self.ground_z,
            z_max=z_max,
            floor_level=0,
            parent_suid=parent_suid,
            metadata={'building_id': building_id}
        )
    
    def _create_floor_volume(
        self,
        building_id: str,
        floor_polygon: Dict,
        ulpin: str,
        parent_suid: str,
        level: int,
        z_base: float,
        height: float
    ) -> VolumetricUnit:
        """Create floor volume"""
        from shapely.wkt import loads
        
        poly = loads(floor_polygon['polygon_wkt'])
        geometry = self._extrude_polygon(poly, z_base, z_base + height)
        
        return VolumetricUnit(
            suid=f"floor-{building_id}-L{level}",
            uln=ulpin,
            label=f"Floor {level}",
            unit_type="floor",
            geometry_wkt=geometry,
            z_min=z_base,
            z_max=z_base + height,
            floor_level=level,
            parent_suid=parent_suid,
            metadata={'building_id': building_id}
        )
    
    def _create_unit_volume(
        self,
        building_id: str,
        unit_polygon: Polygon,
        ulpin: str,
        parent_suid: str,
        level: int,
        z_base: float,
        height: float,
        unit_type: str
    ) -> VolumetricUnit:
        """Create unit/apartment volume"""
        geometry = self._extrude_polygon(unit_polygon, z_base, z_base + height)
        
        return VolumetricUnit(
            suid=f"unit-{building_id}-L{level}-{ulpin[-4:]}",
            uln=ulpin,
            label=f"Unit {ulpin[-4:]}",
            unit_type="unit",
            geometry_wkt=geometry,
            z_min=z_base,
            z_max=z_base + height,
            floor_level=level,
            parent_suid=parent_suid,
            metadata={'building_id': building_id, 'unit_type': unit_type}
        )
    
    def _extrude_polygon(self, polygon: Polygon, z_min: float, z_max: float) -> str:
        """Extrude 2D polygon to 3D POLYHEDRALSURFACE Z"""
        if polygon.is_empty:
            return "POLYHEDRALSURFACE Z EMPTY"
        
        # Get exterior ring coordinates
        ext_coords = list(polygon.exterior.coords)
        if len(ext_coords) < 4:  # Need at least 3 unique points + closing
            return "POLYHEDRALSURFACE Z EMPTY"
        
        # Build faces
        faces = []
        
        # Bottom face (z_min)
        bottom_coords = [(x, y, z_min) for x, y in ext_coords]
        faces.append(self._coords_to_polygon(bottom_coords))
        
        # Top face (z_max) - reversed for correct orientation
        top_coords = [(x, y, z_max) for x, y in reversed(ext_coords)]
        faces.append(self._coords_to_polygon(top_coords))
        
        # Side faces
        for i in range(len(ext_coords) - 1):
            x1, y1 = ext_coords[i]
            x2, y2 = ext_coords[i + 1]
            
            side_coords = [
                (x1, y1, z_min),
                (x2, y2, z_min),
                (x2, y2, z_max),
                (x1, y1, z_max),
                (x1, y1, z_min)
            ]
            faces.append(self._coords_to_polygon(side_coords))
        
        # Interior rings (holes) - create side faces for holes
        for interior in polygon.interiors:
            hole_coords = list(interior.coords)
            if len(hole_coords) < 4:
                continue
            
            # Hole bottom (reversed)
            hole_bottom = [(x, y, z_min) for x, y in reversed(hole_coords)]
            faces.append(self._coords_to_polygon(hole_bottom))
            
            # Hole top
            hole_top = [(x, y, z_max) for x, y in hole_coords]
            faces.append(self._coords_to_polygon(hole_top))
            
            # Hole sides
            for i in range(len(hole_coords) - 1):
                x1, y1 = hole_coords[i]
                x2, y2 = hole_coords[i + 1]
                
                side_coords = [
                    (x1, y1, z_min),
                    (x1, y1, z_max),
                    (x2, y2, z_max),
                    (x2, y2, z_min),
                    (x1, y1, z_min)
                ]
                faces.append(self._coords_to_polygon(side_coords))
        
        # Combine into POLYHEDRALSURFACE
        face_wkts = ', '.join(faces)
        return f"POLYHEDRALSURFACE Z({face_wkts})"
    
    def _coords_to_polygon(self, coords: List[Tuple[float, float, float]]) -> str:
        """Convert 3D coordinates to POLYGON Z WKT"""
        coord_str = ', '.join(f"{x} {y} {z}" for x, y, z in coords)
        return f"POLYGON Z(({coord_str}))"


class PointCloudFusion:
    """Refine 3D volumes using LiDAR point cloud data"""
    
    def __init__(self, voxel_size: float = 0.5):
        self.voxel_size = voxel_size
    
    def refine_building_boundaries(
        self,
        volumetric_units: List[VolumetricUnit],
        building_points: np.ndarray,  # [N, 3] XYZ
        roof_points: np.ndarray = None
    ) -> List[VolumetricUnit]:
        """Refine building footprint using point cloud"""
        from shapely.wkt import loads
        from scipy.spatial import ConvexHull
        
        # Extract building footprint from points
        xy = building_points[:, :2]
        
        if len(xy) < 10:
            return volumetric_units
        
        # Compute concave hull (alpha shape) or convex hull
        try:
            hull = ConvexHull(xy)
            hull_points = xy[hull.vertices]
            refined_footprint = Polygon(hull_points)
        except:
            # Fallback: bounding box with buffer
            minx, miny = xy.min(axis=0)
            maxx, maxy = xy.max(axis=0)
            refined_footprint = box(minx, miny, maxx, maxy).buffer(self.voxel_size)
        
        # Update units with refined footprint
        for unit in volumetric_units:
            if unit.unit_type in ('parcel', 'building', 'floor'):
                geom = loads(unit.geometry_wkt)
                # Replace XY with refined footprint, keep Z
                refined_geom = self._replace_xy(geom, refined_footprint)
                unit.geometry_wkt = refined_geom
        
        return volumetric_units
    
    def detect_floor_heights(
        self,
        points: np.ndarray,
        floor_polygons: List[Dict]
    ) -> List[Dict]:
        """Detect actual floor heights from point cloud"""
        z = points[:, 2]
        
        # Find horizontal slices (peaks in Z histogram)
        hist, bins = np.histogram(z, bins=50)
        
        # Find peaks (floor levels)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(points) * 0.01:
                peaks.append((bins[i] + bins[i+1]) / 2)
        
        # Sort peaks
        peaks.sort()
        
        # Assign to floor polygons
        for fp in floor_polygons:
            level = fp.get('level', 0)
            expected_z = fp.get('z_base', level * 3.0)
            
            # Find closest peak
            if peaks:
                closest = min(peaks, key=lambda p: abs(p - expected_z))
                fp['detected_z_base'] = closest
                if level > 0 and len(peaks) > level:
                    fp['detected_height'] = peaks[level] - peaks[level-1]
        
        return floor_polygons
    
    def _replace_xy(self, geom_wkt: str, new_footprint: Polygon) -> str:
        """Replace XY coordinates in 3D geometry with new footprint"""
        # This is simplified - full implementation would parse POLYHEDRALSURFACE
        # and rebuild with new XY
        return geom_wkt


def create_volumetric_units_from_building(
    building_id: str,
    footprint: Polygon,
    num_floors: int,
    floor_height: float = 3.0,
    ground_z: float = 0.0,
    ulpin_generator=None,
    unit_layout: str = "grid"  # grid|perimeter|custom
) -> List[VolumetricUnit]:
    """Quick generator for synthetic building"""
    if ulpin_generator is None:
        from backend.core.ulpin import ULPINGenerator
        ulpin_generator = ULPINGenerator()
    
    delineator = VerticalDelineator(
        default_floor_height=floor_height,
        ground_z=ground_z
    )
    
    # Create synthetic floor polygons
    floor_polygons = []
    for level in range(num_floors):
        z_base = ground_z + level * floor_height
        
        if unit_layout == "grid":
            # Split into 4 units per floor
            units = _split_polygon_grid(footprint, 2, 2)
        elif unit_layout == "perimeter":
            # Perimeter units with central core
            units = _split_polygon_perimeter(footprint, 4)
        else:
            units = [footprint]
        
        for j, unit_poly in enumerate(units):
            floor_polygons.append({
                'level': level,
                'z_base': z_base,
                'height': floor_height,
                'polygon_wkt': unit_poly.wkt,
                'unit_id': f"U{level}{j}",
                'unit_type': 'residential'
            })
    
    return delineator.extrude_floor_plan(building_id, floor_polygons, ulpin_generator)


def _split_polygon_grid(polygon: Polygon, rows: int, cols: int) -> List[Polygon]:
    """Split polygon into grid"""
    minx, miny, maxx, maxy = polygon.bounds
    width = (maxx - minx) / cols
    height = (maxy - miny) / rows
    
    units = []
    for i in range(rows):
        for j in range(cols):
            cell = box(
                minx + j * width,
                miny + i * height,
                minx + (j + 1) * width,
                miny + (i + 1) * height
            )
            intersection = polygon.intersection(cell)
            if intersection.area > 0:
                units.append(intersection)
    
    return units


def _split_polygon_perimeter(polygon: Polygon, num_units: int) -> List[Polygon]:
    """Split polygon into perimeter units"""
    # Simplified: create buffer rings
    units = []
    centroid = polygon.centroid
    
    for i in range(num_units):
        # Create wedge from centroid
        angle_start = i * 2 * math.pi / num_units
        angle_end = (i + 1) * 2 * math.pi / num_units
        
        # Create triangle from centroid to boundary
        # This is simplified - real implementation uses proper partitioning
        pass
    
    return [polygon] if not units else units