"""
OSM/Overpass API Ingestion for Mumbai Cadastral Data
Fetches building footprints, heights, and attributes from OpenStreetMap
"""
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform
from shapely.wkt import dumps as wkt_dumps
import math

logger = logging.getLogger(__name__)


@dataclass
class OSMBuilding:
    """Building from OSM data"""
    osm_id: int
    name: Optional[str]
    building_type: str  # residential|commercial|mixed|apartments|etc
    levels: Optional[int]
    height: Optional[float]  # meters
    min_height: Optional[float]  # base height
    roof_shape: Optional[str]
    roof_height: Optional[float]
    geometry: Polygon
    centroid: Tuple[float, float]
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_footprint_wkt(self) -> str:
        return wkt_dumps(self.geometry)


@dataclass
class MumbaiBounds:
    """Mumbai bounding box"""
    south: float = 18.85
    west: float = 72.75
    north: float = 19.30
    east: float = 73.10
    
    # Ward-level bounds (approximate)
    WARDS = {
        'A': (72.82, 18.93, 72.87, 18.96),  # Colaba
        'B': (72.81, 18.96, 72.86, 18.99),  # Fort
        'C': (72.80, 18.99, 72.85, 19.02),  # Marine Lines
        'D': (72.82, 18.99, 72.87, 19.02),  # Grant Road
        'E': (72.85, 18.96, 72.90, 19.00),  # Byculla
        'F_N': (72.85, 19.00, 72.90, 19.05), # Matunga
        'F_S': (72.85, 18.96, 72.90, 19.00), # Parel
        'G_N': (72.88, 19.00, 72.93, 19.05), # Dadar
        'G_S': (72.88, 19.00, 72.93, 19.05), # Elphinstone
        'H_E': (72.90, 19.00, 72.95, 19.05), # Bandra
        'H_W': (72.85, 19.00, 72.90, 19.05), # Khar
        'K_E': (72.90, 19.05, 72.95, 19.10), # Andheri East
        'K_W': (72.85, 19.05, 72.90, 19.10), # Andheri West
        'L': (72.80, 19.10, 72.85, 19.15),   # Kurla
        'M_E': (72.90, 19.10, 72.95, 19.15), # Chembur
        'M_W': (72.85, 19.10, 72.90, 19.15), # Ghatkopar
        'N': (72.80, 19.15, 72.85, 19.20),   # Vikhroli
        'P_N': (72.90, 19.15, 72.95, 19.20), # Malad
        'P_S': (72.85, 19.15, 72.90, 19.20), # Goregaon
        'R_C': (72.80, 19.20, 72.85, 19.25), # Borivali
        'R_N': (72.85, 19.20, 72.90, 19.25), # Dahisar
        'R_S': (72.80, 19.20, 72.85, 19.25), # Kandivali
        'S': (72.90, 19.15, 72.95, 19.20),   # Bhandup
        'T': (72.85, 19.20, 72.90, 19.25),   # Mulund
    }


class OverpassClient:
    """Async client for Overpass API"""
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    # Alternative endpoints
    ALTERNATIVES = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    
    def __init__(self, timeout: int = 180):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str) -> Dict:
        """Execute Overpass QL query"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        
        for url in [self.OVERPASS_URL] + self.ALTERNATIVES:
            try:
                async with self.session.post(
                    url,
                    data={'data': query},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"Overpass {url} returned {resp.status}")
            except Exception as e:
                logger.warning(f"Overpass {url} failed: {e}")
        
        raise RuntimeError("All Overpass endpoints failed")
    
    def build_building_query(
        self,
        bbox: Tuple[float, float, float, float],  # west, south, east, north
        include_tags: bool = True
    ) -> str:
        """Build query for buildings in bbox"""
        west, south, east, north = bbox
        
        tag_filter = ""
        if include_tags:
            tag_filter = """
            out body;
            >;
            out skel qt;
            """
        
        return f"""
        [out:json][timeout:180];
        (
          way["building"]({south},{west},{north},{east});
          relation["building"]({south},{west},{north},{east});
        );
        {tag_filter}
        """
    
    def build_ward_query(self, ward: str) -> str:
        """Build query for specific Mumbai ward"""
        bounds = MumbaiBounds.WARDS.get(ward.upper())
        if not bounds:
            raise ValueError(f"Unknown ward: {ward}")
        return self.build_building_query(bounds)


class OSMBuildingParser:
    """Parse OSM elements to building objects"""
    
    BUILDING_TYPE_MAP = {
        'residential': 'residential',
        'apartments': 'residential',
        'house': 'residential',
        'detached': 'residential',
        'semidetached_house': 'residential',
        'terrace': 'residential',
        'commercial': 'commercial',
        'office': 'commercial',
        'retail': 'commercial',
        'shop': 'commercial',
        'supermarket': 'commercial',
        'mall': 'commercial',
        'industrial': 'industrial',
        'warehouse': 'industrial',
        'factory': 'industrial',
        'mixed': 'mixed',
        'mixed_use': 'mixed',
        'hotel': 'hospitality',
        'hostel': 'hospitality',
        'hospital': 'institutional',
        'school': 'institutional',
        'university': 'institutional',
        'civic': 'institutional',
        'government': 'institutional',
        'religious': 'institutional',
        'parking': 'parking',
        'garage': 'parking',
        'roof': 'utility',
        'shed': 'utility',
        'construction': 'construction',
    }
    
    ROOF_SHAPES = {
        'flat': 'flat',
        'gabled': 'gabled',
        'hipped': 'hipped',
        'pyramidal': 'pyramid',
        'dome': 'dome',
        'mansard': 'mansard',
        'gambrel': 'gambrel',
        'skillion': 'skillion',
        'sawtooth': 'sawtooth',
    }
    
    DEFAULT_FLOOR_HEIGHT = 3.0
    DEFAULT_GROUND_HEIGHT = 0.0
    
    def parse(self, geojson: Dict) -> List[OSMBuilding]:
        """Parse Overpass GeoJSON response to building list"""
        buildings = []
        
        # Index nodes by ID
        nodes = {}
        for element in geojson.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = (element['lon'], element['lat'])
        
        # Process ways and relations
        for element in geojson.get('elements', []):
            if element['type'] == 'way':
                building = self._parse_way(element, nodes)
                if building:
                    buildings.append(building)
            elif element['type'] == 'relation':
                building = self._parse_relation(element, nodes, geojson['elements'])
                if building:
                    buildings.append(building)
        
        return buildings
    
    def _parse_way(self, way: Dict, nodes: Dict) -> Optional[OSMBuilding]:
        """Parse OSM way to building"""
        tags = way.get('tags', {})
        
        # Skip if not a building
        if 'building' not in tags:
            return None
        
        # Get geometry
        node_refs = way.get('nodes', [])
        coords = [nodes[nid] for nid in node_refs if nid in nodes]
        
        if len(coords) < 4:  # Need at least 3 unique + closing
            return None
        
        # Check if closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            polygon = Polygon(coords)
            if not polygon.is_valid or polygon.area < 5:  # Min 5 m²
                return None
        except Exception:
            return None
        
        # Extract attributes
        osm_id = way['id']
        name = tags.get('name')
        building_tag = tags.get('building', 'yes')
        building_type = self.BUILDING_TYPE_MAP.get(building_tag, 'unknown')
        
        # Height/levels
        levels = self._parse_int(tags.get('building:levels') or tags.get('levels'))
        height = self._parse_float(tags.get('height') or tags.get('building:height'))
        min_height = self._parse_float(tags.get('min_height') or tags.get('building:min_height'))
        
        # Roof
        roof_shape = self.ROOF_SHAPES.get(tags.get('roof:shape', '').lower())
        roof_height = self._parse_float(tags.get('roof:height'))
        
        # Estimate missing values
        if levels is None and height is not None:
            levels = max(1, round(height / self.DEFAULT_FLOOR_HEIGHT))
        elif height is None and levels is not None:
            height = levels * self.DEFAULT_FLOOR_HEIGHT
        elif height is None and levels is None:
            # Default based on building type
            levels = self._estimate_levels(building_type, polygon.area)
            height = levels * self.DEFAULT_FLOOR_HEIGHT
        
        if min_height is None:
            min_height = self.DEFAULT_GROUND_HEIGHT
        
        centroid = (polygon.centroid.x, polygon.centroid.y)
        
        return OSMBuilding(
            osm_id=osm_id,
            name=name,
            building_type=building_type,
            levels=levels,
            height=height,
            min_height=min_height,
            roof_shape=roof_shape,
            roof_height=roof_height,
            geometry=polygon,
            centroid=centroid,
            tags=tags
        )
    
    def _parse_relation(self, relation: Dict, nodes: Dict, all_elements: List) -> Optional[OSMBuilding]:
        """Parse OSM relation (multipolygon) to building"""
        tags = relation.get('tags', {})
        
        if tags.get('type') != 'multipolygon' or 'building' not in tags:
            return None
        
        # Build member ways
        outer_rings = []
        inner_rings = []
        
        for member in relation.get('members', []):
            if member['type'] != 'way' or member['role'] not in ('outer', 'inner'):
                continue
            
            # Find the way element
            way_elem = next((e for e in all_elements if e['type'] == 'way' and e['id'] == member['ref']), None)
            if not way_elem:
                continue
            
            node_refs = way_elem.get('nodes', [])
            coords = [nodes[nid] for nid in node_refs if nid in nodes]
            
            if len(coords) < 4:
                continue
            
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            if member['role'] == 'outer':
                outer_rings.append(coords)
            else:
                inner_rings.append(coords)
        
        if not outer_rings:
            return None
        
        try:
            # Create polygon with holes
            polygon = Polygon(outer_rings[0], inner_rings[1:] if len(inner_rings) > 1 else inner_rings)
            if not polygon.is_valid or polygon.area < 5:
                return None
        except Exception:
            return None
        
        # Similar attribute extraction as way
        osm_id = relation['id']
        name = tags.get('name')
        building_tag = tags.get('building', 'yes')
        building_type = self.BUILDING_TYPE_MAP.get(building_tag, 'unknown')
        
        levels = self._parse_int(tags.get('building:levels') or tags.get('levels'))
        height = self._parse_float(tags.get('height') or tags.get('building:height'))
        min_height = self._parse_float(tags.get('min_height') or tags.get('building:min_height'))
        
        roof_shape = self.ROOF_SHAPES.get(tags.get('roof:shape', '').lower())
        roof_height = self._parse_float(tags.get('roof:height'))
        
        if levels is None and height is not None:
            levels = max(1, round(height / self.DEFAULT_FLOOR_HEIGHT))
        elif height is None and levels is not None:
            height = levels * self.DEFAULT_FLOOR_HEIGHT
        elif height is None and levels is None:
            levels = self._estimate_levels(building_type, polygon.area)
            height = levels * self.DEFAULT_FLOOR_HEIGHT
        
        if min_height is None:
            min_height = self.DEFAULT_GROUND_HEIGHT
        
        centroid = (polygon.centroid.x, polygon.centroid.y)
        
        return OSMBuilding(
            osm_id=osm_id,
            name=name,
            building_type=building_type,
            levels=levels,
            height=height,
            min_height=min_height,
            roof_shape=roof_shape,
            roof_height=roof_height,
            geometry=polygon,
            centroid=centroid,
            tags=tags
        )
    
    def _parse_int(self, val: Optional[str]) -> Optional[int]:
        try:
            return int(float(val)) if val else None
        except (ValueError, TypeError):
            return None
    
    def _parse_float(self, val: Optional[str]) -> Optional[float]:
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None
    
    def _estimate_levels(self, building_type: str, area: float) -> int:
        """Estimate floor count from building type and footprint area"""
        if building_type == 'residential':
            if area < 100: return 1
            elif area < 300: return 2
            elif area < 800: return 4
            elif area < 2000: return 8
            else: return 12
        elif building_type == 'commercial':
            if area < 200: return 1
            elif area < 500: return 2
            elif area < 1500: return 4
            else: return 10
        elif building_type == 'industrial':
            return 1
        elif building_type == 'mixed':
            if area < 500: return 3
            elif area < 1500: return 6
            else: return 15
        elif building_type == 'hospitality':
            return max(3, round(math.sqrt(area) / 10))
        elif building_type == 'institutional':
            return max(2, round(math.sqrt(area) / 15))
        else:
            return max(1, round(math.sqrt(area) / 20))


class MumbaiOSMIngestion:
    """Complete OSM ingestion pipeline for Mumbai"""
    
    def __init__(
        self,
        output_dir: str = "data/mumbai_osm",
        default_floor_height: float = 3.0,
        ground_z: float = 0.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_floor_height = default_floor_height
        self.ground_z = ground_z
        self.client = OverpassClient()
        self.parser = OSMBuildingParser()
    
    async def fetch_ward(self, ward: str) -> List[OSMBuilding]:
        """Fetch buildings for a specific Mumbai ward"""
        query = self.client.build_ward_query(ward)
        async with self.client as client:
            result = await client.query(query)
        return self.parser.parse(result)
    
    async def fetch_bbox(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> List[OSMBuilding]:
        """Fetch buildings for arbitrary bbox"""
        query = self.client.build_building_query(bbox)
        async with self.client as client:
            result = await client.query(query)
        return self.parser.parse(result)
    
    async def fetch_all_mumbai(self) -> Dict[str, List[OSMBuilding]]:
        """Fetch all wards in parallel"""
        wards = list(MumbaiBounds.WARDS.keys())
        tasks = [self.fetch_ward(ward) for ward in wards]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        ward_buildings = {}
        for ward, result in zip(wards, results):
            if isinstance(result, Exception):
                logger.error(f"Ward {ward} failed: {result}")
                ward_buildings[ward] = []
            else:
                ward_buildings[ward] = result
                logger.info(f"Ward {ward}: {len(result)} buildings")
        
        return ward_buildings
    
    def buildings_to_geojson(self, buildings: List[OSMBuilding]) -> Dict:
        """Convert buildings to GeoJSON FeatureCollection"""
        features = []
        for b in buildings:
            features.append({
                "type": "Feature",
                "id": str(b.osm_id),
                "geometry": json.loads(wkt_dumps(b.geometry)),
                "properties": {
                    "osm_id": b.osm_id,
                    "name": b.name,
                    "building_type": b.building_type,
                    "levels": b.levels,
                    "height": b.height,
                    "min_height": b.min_height,
                    "roof_shape": b.roof_shape,
                    "roof_height": b.roof_height,
                    "centroid": b.centroid,
                    "tags": b.tags
                }
            })
        return {"type": "FeatureCollection", "features": features}
    
    def buildings_to_floor_polygons(self, buildings: List[OSMBuilding]) -> List[Dict]:
        """Convert OSM buildings to floor polygon format for vertical delineation"""
        floor_polygons = []
        
        for building in buildings:
            if building.levels is None or building.levels <= 0:
                continue
            
            # Use min_height as ground base
            z_base = building.min_height or self.ground_z
            floor_height = (building.height or (building.levels * self.default_floor_height)) / building.levels
            
            # Determine unit layout based on building type and size
            unit_layout = self._determine_unit_layout(building)
            units_per_floor = self._estimate_units_per_floor(building)
            
            for level in range(building.levels):
                level_z = z_base + level * floor_height
                
                floor_polygons.append({
                    'level': level,
                    'z_base': level_z,
                    'height': floor_height,
                    'polygon_wkt': building.to_footprint_wkt(),
                    'unit_id': f"U{level:02d}",
                    'unit_type': building.building_type,
                    'unit_layout': unit_layout,
                    'units_per_floor': units_per_floor,
                    'building_id': str(building.osm_id),
                    'building_name': building.name,
                    'roof_shape': building.roof_shape,
                    'roof_height': building.roof_height
                })
        
        return floor_polygons
    
    def _area_sqm(self, building: OSMBuilding) -> float:
        """Calculate building footprint area in square meters"""
        centroid = building.geometry.centroid
        lat = centroid.y
        deg_lat_m = 111000.0
        deg_lon_m = 111000.0 * math.cos(math.radians(lat))
        return building.geometry.area * deg_lat_m * deg_lon_m
    
    def _determine_unit_layout(self, building: OSMBuilding) -> str:
        """Determine unit layout based on building type and footprint"""
        area_sqm = self._area_sqm(building)
        
        if building.building_type in ('apartments', 'residential'):
            if area_sqm > 800:
                return 'corridor'  # Large apartment blocks often have corridor layout
            elif area_sqm > 300:
                return 'perimeter'  # Medium buildings with units around core
            else:
                return 'grid'  # Small buildings
        elif building.building_type == 'commercial':
            if area_sqm > 1500:
                return 'corridor'
            else:
                return 'grid'
        elif building.building_type == 'mixed':
            return 'perimeter'
        else:
            return 'grid'
    
    def _estimate_units_per_floor(self, building: OSMBuilding) -> int:
        """Estimate units per floor based on building type and area"""
        area_sqm = self._area_sqm(building)
        
        if building.building_type in ('apartments', 'residential'):
            # ~70-100 m² per flat for residential
            return max(1, round(area_sqm / 85))
        elif building.building_type == 'commercial':
            # Larger commercial units
            return max(1, round(area_sqm / 300))
        elif building.building_type == 'mixed':
            return max(2, round(area_sqm / 150))
        else:
            return max(1, round(area_sqm / 100))
    
    def save_geojson(self, buildings: List[OSMBuilding], filename: str):
        """Save buildings to GeoJSON file"""
        geojson = self.buildings_to_geojson(buildings)
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f"Saved {len(buildings)} buildings to {filepath}")
    
    def save_floor_polygons(self, buildings: List[OSMBuilding], filename: str):
        """Save floor polygons for vertical delineation"""
        floor_polys = self.buildings_to_floor_polygons(buildings)
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(floor_polys, f, indent=2)
        logger.info(f"Saved {len(floor_polys)} floor polygons to {filepath}")


async def main():
    """Demo: Fetch buildings for a Mumbai ward"""
    import sys
    logging.basicConfig(level=logging.INFO)
    
    ingestion = MumbaiOSMIngestion()
    
    # Fetch a single ward (smaller test)
    print("Fetching Ward A (Colaba)...")
    buildings = await ingestion.fetch_ward('A')
    print(f"Found {len(buildings)} buildings")
    
    if buildings:
        # Save outputs
        ingestion.save_geojson(buildings, "ward_a_buildings.geojson")
        ingestion.save_floor_polygons(buildings, "ward_a_floors.json")
        
        # Print summary
        for b in buildings[:5]:
            print(f"  OSM:{b.osm_id} {b.name or 'unnamed'} - {b.building_type} - {b.levels}L - {b.height}m")


if __name__ == "__main__":
    asyncio.run(main())