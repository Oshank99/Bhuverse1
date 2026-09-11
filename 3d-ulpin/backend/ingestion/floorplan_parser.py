"""
Floor Plan Ingestion - DXF/DWG and IFC parsers
Converts CAD/BIM to 3D spatial units
"""
import ezdxf
from ezdxf.math import Vec3
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import math


@dataclass
class FloorPolygon:
    """Single floor polygon with metadata"""
    level: int          # Floor level (0=ground, -1=basement, etc.)
    height: float       # Floor-to-ceiling height (m)
    z_base: float       # Base elevation (m)
    polygon_wkt: str    # 2D polygon WKT
    unit_id: str        # Unit identifier (e.g., "Apt-3B")
    unit_type: str      # residential|commercial|parking|utility|common
    area: float         # Floor area (m2)
    metadata: Dict = field(default_factory=dict)


@dataclass
class BuildingFloorPlan:
    """Complete building floor plan"""
    building_id: str
    floors: List[FloorPolygon]
    crs: str = "EPSG:4326"
    metadata: Dict = field(default_factory=dict)


class DXFParser:
    """Parse DXF/DWG floor plans to 3D polygons"""
    
    def __init__(self, default_floor_height: float = 3.0):
        self.default_floor_height = default_floor_height
        self.layer_mapping = {
            'A-WALL': 'wall',
            'A-DOOR': 'door',
            'A-WIND': 'window',
            'A-FLOR': 'floor_boundary',
            'A-UNIT': 'unit_boundary',
            'A-TEXT': 'annotation',
            'A-DIMS': 'dimension'
        }
    
    def parse(self, file_path: str) -> BuildingFloorPlan:
        """Parse DXF file to building floor plan"""
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # Detect layers for floor boundaries
        floor_layers = self._detect_floor_layers(doc)
        
        floors = []
        for layer_name, level_info in floor_layers.items():
            level = level_info['level']
            z_base = level_info['z_base']
            height = level_info.get('height', self.default_floor_height)
            
            # Extract polygons from layer
            polygons = self._extract_polygons_from_layer(msp, layer_name)
            
            for poly in polygons:
                unit_id, unit_type = self._identify_unit(poly, msp, layer_name)
                area = self._compute_area(poly)
                
                if area > 1.0:  # Ignore tiny polygons
                    floors.append(FloorPolygon(
                        level=level,
                        height=height,
                        z_base=z_base,
                        polygon_wkt=self._polygon_to_wkt(poly),
                        unit_id=unit_id,
                        unit_type=unit_type,
                        area=area,
                        metadata={'layer': layer_name, 'source': 'dxf'}
                    ))
        
        # Group by level and merge if needed
        floors = self._merge_floor_polygons(floors)
        
        building_id = Path(file_path).stem
        return BuildingFloorPlan(
            building_id=building_id,
            floors=floors,
            metadata={'source_file': file_path, 'parser': 'dxf'}
        )
    
    def _detect_floor_layers(self, doc) -> Dict:
        """Detect floor levels from layer names"""
        # Common patterns: "FL_01", "FLOOR_1", "LEVEL_0", "BASEMENT", etc.
        floor_layers = {}
        
        for layer in doc.layers:
            name = layer.dxf.name.upper()
            level = None
            z_base = 0.0
            
            # Pattern matching
            if 'BASEMENT' in name or 'B-' in name:
                level = -1
                z_base = -3.0
            elif 'GROUND' in name or 'GF' in name or 'FL_00' in name:
                level = 0
                z_base = 0.0
            elif 'ROOF' in name:
                level = 99
                z_base = 0.0  # Will be computed
            else:
                # Try numeric patterns
                import re
                match = re.search(r'(FL|FLOOR|LEVEL|L)[_\-]?(\d+)', name)
                if match:
                    level = int(match.group(2))
                    z_base = level * self.default_floor_height
            
            if level is not None:
                floor_layers[layer.dxf.name] = {
                    'level': level,
                    'z_base': z_base,
                    'height': self.default_floor_height
                }
        
        return floor_layers
    
    def _extract_polygons_from_layer(self, msp, layer_name: str) -> List[List[Vec3]]:
        """Extract closed polygons (LWPOLYLINE, POLYLINE) from layer"""
        polygons = []
        
        for entity in msp.query(f'*[layer=="{layer_name}"]'):
            if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                if entity.closed or self._is_closed(entity):
                    points = self._get_polyline_points(entity)
                    if len(points) >= 3:
                        polygons.append(points)
            elif entity.dxftype() == 'HATCH':
                # HATCH boundaries
                for path in entity.paths:
                    if path.path_type_flags == 1:  # External
                        points = [Vec3(v[0], v[1]) for v in path.vertices]
                        if len(points) >= 3:
                            polygons.append(points)
        
        return polygons
    
    def _get_polyline_points(self, entity) -> List[Vec3]:
        """Get vertices from polyline"""
        if entity.dxftype() == 'LWPOLYLINE':
            return [Vec3(p[0], p[1]) for p in entity.get_points()]
        else:
            return [Vec3(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
    
    def _is_closed(self, entity) -> bool:
        """Check if polyline is closed"""
        if hasattr(entity, 'closed'):
            return entity.closed
        points = self._get_polyline_points(entity)
        return points[0].distance(points[-1]) < 0.001
    
    def _identify_unit(self, polygon: List[Vec3], msp, layer_name: str) -> Tuple[str, str]:
        """Identify unit ID and type from text/annotations inside polygon"""
        # Find text entities inside polygon
        centroid = self._polygon_centroid(polygon)
        
        unit_id = "UNIT"
        unit_type = "residential"
        
        for text in msp.query('TEXT MTEXT'):
            if text.dxf.layer == layer_name:
                pos = Vec3(text.dxf.insert.x, text.dxf.insert.y)
                if self._point_in_polygon(pos, polygon):
                    content = text.plain_text() if hasattr(text, 'plain_text') else text.dxf.text
                    # Parse unit info from text
                    unit_id, unit_type = self._parse_unit_text(content)
                    break
        
        return unit_id, unit_type
    
    def _parse_unit_text(self, text: str) -> Tuple[str, str]:
        """Parse unit identifier and type from annotation text"""
        text = text.strip().upper()
        unit_type = "residential"
        
        if any(kw in text for kw in ['SHOP', 'RETAIL', 'COMMERCIAL', 'OFFICE']):
            unit_type = "commercial"
        elif any(kw in text for kw in ['PARKING', 'GARAGE', 'CAR']):
            unit_type = "parking"
        elif any(kw in text for kw in ['UTILITY', 'MECH', 'ELEC', 'HVAC']):
            unit_type = "utility"
        elif any(kw in text for kw in ['COMMON', 'LOBBY', 'CORRIDOR', 'STAIR', 'ELEV']):
            unit_type = "common"
        
        # Extract unit ID (alphanumeric)
        import re
        match = re.search(r'([A-Z]\d+|[A-Z]-\d+|\d+[A-Z]?)', text)
        unit_id = match.group(1) if match else text[:20]
        
        return unit_id, unit_type
    
    def _polygon_centroid(self, polygon: List[Vec3]) -> Vec3:
        """Compute polygon centroid"""
        x = sum(p.x for p in polygon) / len(polygon)
        y = sum(p.y for p in polygon) / len(polygon)
        return Vec3(x, y)
    
    def _point_in_polygon(self, point: Vec3, polygon: List[Vec3]) -> bool:
        """Ray casting point-in-polygon test"""
        x, y = point.x, point.y
        inside = False
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside
    
    def _compute_area(self, polygon: List[Vec3]) -> float:
        """Shoelace formula for polygon area"""
        area = 0.0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i].x * polygon[j].y
            area -= polygon[j].x * polygon[i].y
        return abs(area) / 2.0
    
    def _polygon_to_wkt(self, polygon: List[Vec3]) -> str:
        """Convert polygon to WKT"""
        coords = ', '.join(f"{p.x} {p.y}" for p in polygon)
        coords += f", {polygon[0].x} {polygon[0].y}"  # Close ring
        return f"POLYGON(({coords}))"
    
    def _merge_floor_polygons(self, floors: List[FloorPolygon]) -> List[FloorPolygon]:
        """Merge adjacent polygons on same floor with same unit"""
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        
        grouped = {}
        for fp in floors:
            key = (fp.level, fp.unit_id, fp.unit_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(fp)
        
        merged = []
        for (level, unit_id, unit_type), fps in grouped.items():
            if len(fps) == 1:
                merged.append(fps[0])
            else:
                # Union polygons
                shapely_polys = [Polygon(fp.polygon_wkt) for fp in fps]
                union = unary_union(shapely_polys)
                
                if union.geom_type == 'Polygon':
                    polys = [union]
                else:
                    polys = list(union.geoms)
                
                for poly in polys:
                    if poly.area > 1.0:
                        coords = list(poly.exterior.coords)
                        wkt = "POLYGON((" + ", ".join(f"{x} {y}" for x, y in coords) + "))"
                        merged.append(FloorPolygon(
                            level=level,
                            height=fps[0].height,
                            z_base=fps[0].z_base,
                            polygon_wkt=wkt,
                            unit_id=unit_id,
                            unit_type=unit_type,
                            area=poly.area,
                            metadata={'merged': True, 'source': 'dxf'}
                        ))
        
        return merged


class IFCParser:
    """Parse IFC (BIM) to semantic 3D building elements"""
    
    def __init__(self):
        try:
            import ifcopenshell
            self.ifcopenshell = ifcopenshell
        except ImportError:
            self.ifcopenshell = None
    
    def parse(self, file_path: str) -> Dict:
        """Parse IFC file to building elements"""
        if self.ifcopenshell is None:
            raise RuntimeError("ifcopenshell not installed")
        
        model = self.ifcopenshell.open(file_path)
        
        result = {
            'building_id': Path(file_path).stem,
            'storeys': [],
            'spaces': [],
            'elements': [],
            'metadata': {'source_file': file_path, 'parser': 'ifc'}
        }
        
        # Get building storeys
        for storey in model.by_type('IfcBuildingStorey'):
            elevation = self._get_elevation(storey)
            result['storeys'].append({
                'global_id': storey.GlobalId,
                'name': storey.Name,
                'elevation': elevation,
                'height': self._get_storey_height(storey, model)
            })
        
        # Get spaces (rooms, apartments)
        for space in model.by_type('IfcSpace'):
            storey = self._get_containing_storey(space, model)
            geometry = self._get_space_geometry(space)
            
            result['spaces'].append({
                'global_id': space.GlobalId,
                'name': space.Name,
                'long_name': space.LongName,
                'storey': storey,
                'geometry': geometry,
                'area': self._get_area(space),
                'volume': self._get_volume(space)
            })
        
        # Get building elements (walls, slabs, doors, windows)
        for element in model.by_type('IfcBuildingElement'):
            if element.is_a() in ('IfcWall', 'IfcSlab', 'IfcDoor', 'IfcWindow'):
                result['elements'].append({
                    'global_id': element.GlobalId,
                    'type': element.is_a(),
                    'name': element.Name,
                    'geometry': self._get_element_geometry(element),
                    'storey': self._get_containing_storey(element, model)
                })
        
        return result
    
    def _get_elevation(self, storey) -> float:
        """Get storey elevation"""
        if hasattr(storey, 'Elevation'):
            return storey.Elevation
        # Compute from object placement
        if storey.ObjectPlacement:
            return self._get_placement_z(storey.ObjectPlacement)
        return 0.0
    
    def _get_storey_height(self, storey, model) -> float:
        """Get storey height"""
        # Try to find next storey elevation
        all_storeys = model.by_type('IfcBuildingStorey')
        elevations = [(s, self._get_elevation(s)) for s in all_storeys]
        elevations.sort(key=lambda x: x[1])
        
        for i, (s, e) in enumerate(elevations):
            if s.GlobalId == storey.GlobalId:
                if i + 1 < len(elevations):
                    return elevations[i+1][1] - e
                return 3.0  # Default
        return 3.0
    
    def _get_containing_storey(self, element, model) -> Optional[str]:
        """Find storey containing element"""
        for rel in model.by_type('IfcRelContainedInSpatialStructure'):
            if element.GlobalId in [e.GlobalId for e in rel.RelatedElements]:
                return rel.RelatingStructure.GlobalId
        return None
    
    def _get_placement_z(self, placement) -> float:
        """Extract Z from IfcObjectPlacement"""
        if placement.RelativePlacement:
            loc = placement.RelativePlacement.Location
            if loc and loc.Coordinates:
                return loc.Coordinates[2] if len(loc.Coordinates) > 2 else 0.0
        return 0.0
    
    def _get_space_geometry(self, space) -> Optional[str]:
        """Get space geometry as WKT"""
        # Simplified - use bounding box
        if space.ObjectPlacement:
            # Would need proper geometry traversal
            return None
        return None
    
    def _get_element_geometry(self, element) -> Optional[str]:
        """Get element geometry as WKT"""
        return None
    
    def _get_area(self, space) -> float:
        """Get space area from quantity sets"""
        for rel in space.IsDefinedBy:
            if rel.is_a('IfcRelDefinesByProperties'):
                qset = rel.RelatingPropertyDefinition
                if qset.is_a('IfcElementQuantity'):
                    for qty in qset.Quantities:
                        if qty.Name == 'NetFloorArea':
                            return qty.AreaValue
        return 0.0
    
    def _get_volume(self, space) -> float:
        """Get space volume from quantity sets"""
        for rel in space.IsDefinedBy:
            if rel.is_a('IfcRelDefinesByProperties'):
                qset = rel.RelatingPropertyDefinition
                if qset.is_a('IfcElementQuantity'):
                    for qty in qset.Quantities:
                        if qty.Name == 'NetVolume':
                            return qty.VolumeValue
        return 0.0


def parse_floor_plan(file_path: str) -> BuildingFloorPlan:
    """Auto-detect format and parse"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix in ('.dxf', '.dwg'):
        parser = DXFParser()
        return parser.parse(file_path)
    elif suffix == '.ifc':
        parser = IFCParser()
        ifc_data = parser.parse(file_path)
        # Convert to BuildingFloorPlan
        floors = []
        for space in ifc_data['spaces']:
            if space['geometry']:
                floors.append(FloorPolygon(
                    level=0,  # Would map from storey
                    height=3.0,
                    z_base=0.0,
                    polygon_wkt=space['geometry'],
                    unit_id=space['name'] or space['global_id'][:8],
                    unit_type='residential',
                    area=space['area'],
                    metadata={'source': 'ifc', 'global_id': space['global_id']}
                ))
        return BuildingFloorPlan(
            building_id=ifc_data['building_id'],
            floors=floors,
            metadata=ifc_data['metadata']
        )
    else:
        raise ValueError(f"Unsupported format: {suffix}")