"""
3D ULPIN Generator - Deterministic hierarchical identification
Format: [State(2)][District(3)][SubDist(2)][Village(4)][Parcel(6)][Vertical(4)][Unit(4)] = 25 digits
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib
import struct


@dataclass
class ULPINComponents:
    state: str        # 2 digits
    district: str     # 3 digits
    subdistrict: str  # 2 digits
    village: str      # 4 digits
    parcel: str       # 6 digits
    vertical: str     # 4 digits (floor level + sub-level)
    unit: str         # 4 digits (unit identifier)
    
    def __post_init__(self):
        # Validate and pad
        self.state = self._pad(self.state, 2)
        self.district = self._pad(self.district, 3)
        self.subdistrict = self._pad(self.subdistrict, 2)
        self.village = self._pad(self.village, 4)
        self.parcel = self._pad(self.parcel, 6)
        self.vertical = self._pad(self.vertical, 4)
        self.unit = self._pad(self.unit, 4)
    
    @staticmethod
    def _pad(value: str, length: int) -> str:
        return str(value).zfill(length)[-length:]
    
    def to_string(self) -> str:
        return f"{self.state}{self.district}{self.subdistrict}{self.village}{self.parcel}{self.vertical}{self.unit}"
    
    @classmethod
    def from_string(cls, ulpin: str) -> 'ULPINComponents':
        if len(ulpin) != 25:
            raise ValueError(f"Invalid ULPIN length: {len(ulpin)}, expected 25")
        return cls(
            state=ulpin[0:2],
            district=ulpin[2:5],
            subdistrict=ulpin[5:7],
            village=ulpin[7:11],
            parcel=ulpin[11:17],
            vertical=ulpin[17:21],
            unit=ulpin[21:25]
        )
    
    def get_hierarchy_path(self) -> list[str]:
        """Return ULPIN prefixes for each hierarchy level"""
        base = f"{self.state}{self.district}{self.subdistrict}{self.village}{self.parcel}"
        return [
            base + "00000000",  # Parcel level
            base + self.vertical + "0000",  # Floor level
            self.to_string()    # Unit level
        ]


class ULPINGenerator:
    """Deterministic 3D ULPIN generator using coordinate hashing"""
    
    # Floor height assumption (meters)
    FLOOR_HEIGHT = 3.0
    BASEMENT_THRESHOLD = -1.5  # Below this = basement
    
    def __init__(self, admin_hierarchy: dict = None):
        """
        admin_hierarchy: {
            'state': '01',
            'district': '001', 
            'subdistrict': '01',
            'village': '0001'
        }
        """
        self.admin = admin_hierarchy or {
            'state': '01',
            'district': '001',
            'subdistrict': '01', 
            'village': '0001'
        }
        self._parcel_counter = 0
        self._unit_counters = {}  # (parcel, vertical) -> counter
    
    def generate_parcel_ulpin(self, x: float, y: float) -> str:
        """Generate parcel-level ULPIN from coordinates (deterministic)"""
        # Use coordinate hash for deterministic parcel ID
        coord_hash = self._hash_coords(x, y, 6)
        parcel_id = str(coord_hash).zfill(6)
        
        return ULPINComponents(
            state=self.admin['state'],
            district=self.admin['district'],
            subdistrict=self.admin['subdistrict'],
            village=self.admin['village'],
            parcel=parcel_id,
            vertical='0000',  # Parcel level
            unit='0000'
        ).to_string()
    
    def generate_building_ulpin(self, parcel_ulpin: str, building_idx: int = 1) -> str:
        """Generate building ULPIN (uses parcel + building index as vertical)"""
        comps = ULPINComponents.from_string(parcel_ulpin)
        vertical = str(building_idx).zfill(4)
        
        return ULPINComponents(
            state=comps.state,
            district=comps.district,
            subdistrict=comps.subdistrict,
            village=comps.village,
            parcel=comps.parcel,
            vertical=vertical,
            unit='0000'
        ).to_string()
    
    def generate_floor_ulpin(self, building_ulpin: str, floor_level: int) -> str:
        """Generate floor ULPIN from building + floor level"""
        comps = ULPINComponents.from_string(building_ulpin)
        # Vertical: floor level encoded (basement negative, ground=0)
        vertical = self._encode_vertical(floor_level, 0)
        
        return ULPINComponents(
            state=comps.state,
            district=comps.district,
            subdistrict=comps.subdistrict,
            village=comps.village,
            parcel=comps.parcel,
            vertical=vertical,
            unit='0000'
        ).to_string()
    
    def generate_unit_ulpin(
        self, 
        floor_ulpin: str, 
        unit_idx: int,
        x: float = None, 
        y: float = None, 
        z: float = None
    ) -> str:
        """Generate unit/apartment ULPIN"""
        comps = ULPINComponents.from_string(floor_ulpin)
        
        # Use unit index for deterministic unique unit ID
        # Coordinates can be same for multiple units on same floor
        unit_id = str(unit_idx).zfill(4)
        
        return ULPINComponents(
            state=comps.state,
            district=comps.district,
            subdistrict=comps.subdistrict,
            village=comps.village,
            parcel=comps.parcel,
            vertical=comps.vertical,
            unit=unit_id
        ).to_string()
    
    def generate_from_geometry(
        self, 
        geom_wkt: str, 
        level_type: str = 'unit',  # parcel|building|floor|unit
        parent_ulpin: str = None,
        floor_level: int = 0,
        unit_index: int = 1
    ) -> str:
        """Generate ULPIN from geometry centroid"""
        # Parse centroid from WKT (simplified - in production use PostGIS)
        centroid = self._extract_centroid(geom_wkt)
        x, y, z = centroid[0], centroid[1], centroid[2] if len(centroid) > 2 else 0
        
        if level_type == 'parcel':
            return self.generate_parcel_ulpin(x, y)
        elif level_type == 'building':
            if not parent_ulpin:
                parent_ulpin = self.generate_parcel_ulpin(x, y)
            return self.generate_building_ulpin(parent_ulpin, unit_index)
        elif level_type == 'floor':
            if not parent_ulpin:
                parent_ulpin = self.generate_parcel_ulpin(x, y)
                parent_ulpin = self.generate_building_ulpin(parent_ulpin, 1)
            return self.generate_floor_ulpin(parent_ulpin, floor_level)
        elif level_type == 'unit':
            if not parent_ulpin:
                parent_ulpin = self.generate_parcel_ulpin(x, y)
                parent_ulpin = self.generate_building_ulpin(parent_ulpin, 1)
                parent_ulpin = self.generate_floor_ulpin(parent_ulpin, floor_level)
            return self.generate_unit_ulpin(parent_ulpin, unit_index, x, y, z)
        
        raise ValueError(f"Unknown level_type: {level_type}")
    
    def _encode_vertical(self, floor_level: int, sub_level: int) -> str:
        """Encode floor level and sub-level into 4 digits"""
        # Floor: signed 3 digits (-99 to +99), sub-level: 1 digit
        floor_encoded = floor_level + 100  # Offset to make positive (0-199)
        return f"{floor_encoded:03d}{sub_level}"
    
    def _decode_vertical(self, vertical: str) -> Tuple[int, int]:
        """Decode vertical component to floor_level and sub_level"""
        floor_encoded = int(vertical[:3])
        sub_level = int(vertical[3])
        floor_level = floor_encoded - 100
        return floor_level, sub_level
    
    def get_floor_level(self, ulpin: str) -> int:
        """Extract floor level from ULPIN"""
        comps = ULPINComponents.from_string(ulpin)
        floor_level, _ = self._decode_vertical(comps.vertical)
        return floor_level
    
    def _hash_coords(self, x: float, y: float, z: float = 0, digits: int = 6) -> int:
        """Deterministic hash from coordinates"""
        # Quantize to mm precision for stability
        quantized = struct.pack('ddd', round(x, 3), round(y, 3), round(z, 3))
        hash_val = int(hashlib.md5(quantized).hexdigest(), 16)
        return hash_val % (10 ** digits)
    
    def _extract_centroid(self, wkt: str) -> tuple:
        """Simple WKT centroid extraction (use PostGIS in production)"""
        # Handle POLYGON, POLYHEDRALSURFACE, etc.
        import re
        coords = re.findall(r'([\d\.-]+)\s+([\d\.-]+)(?:\s+([\d\.-]+))?', wkt)
        if not coords:
            return (0.0, 0.0, 0.0)
        
        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]
        zs = [float(c[2]) if c[2] else 0.0 for c in coords]
        
        return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))


# Global instance
_default_generator = None

def get_ulpin_generator(admin_hierarchy: dict = None) -> ULPINGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = ULPINGenerator(admin_hierarchy)
    return _default_generator


def generate_3d_ulpin(
    x: float, y: float, z: float = 0,
    level: str = 'unit',
    parent_ulpin: str = None,
    floor_level: int = 0,
    unit_index: int = 1,
    admin: dict = None
) -> str:
    """Convenience function for quick ULPIN generation"""
    gen = get_ulpin_generator(admin)
    return gen.generate_from_geometry(
        f"POINT({x} {y} {z})",
        level_type=level,
        parent_ulpin=parent_ulpin,
        floor_level=floor_level,
        unit_index=unit_index
    )