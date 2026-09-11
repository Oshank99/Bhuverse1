"""
3D Topology Validation using TEN (Tetrahedral Network) model
Detects gaps, overlaps, and validates 3D cadastral topology
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
from enum import Enum
import math


class TopologyRelation(Enum):
    ADJACENT = "adjacent"
    OVERLAP = "overlap"
    GAP = "gap"
    CONTAIN = "contain"
    TOUCH = "touch"
    DISJOINT = "disjoint"


@dataclass
class ValidationResult:
    suid_1: str
    suid_2: str
    relation: TopologyRelation
    details: str
    severity: str  # error|warning|info
    shared_face_wkt: Optional[str] = None


@dataclass
class Tetrahedron:
    """Tetrahedron for TEN model"""
    vertices: List[Tuple[float, float, float]]  # 4 vertices
    suid: str
    face_indices: List[Tuple[int, int, int]] = None  # 4 faces, each 3 vertices
    
    def __post_init__(self):
        if self.face_indices is None:
            # Standard tetrahedron faces (each face omits one vertex)
            self.face_indices = [
                (1, 2, 3),  # face opposite vertex 0
                (0, 2, 3),  # face opposite vertex 1
                (0, 1, 3),  # face opposite vertex 2
                (0, 1, 2),  # face opposite vertex 3
            ]
    
    def get_faces(self) -> List[Tuple[Tuple[float, float, float], ...]]:
        """Get triangular faces as vertex tuples"""
        faces = []
        for idx in self.face_indices:
            face = tuple(self.vertices[i] for i in idx)
            faces.append(face)
        return faces
    
    def volume(self) -> float:
        """Signed volume of tetrahedron"""
        v0, v1, v2, v3 = self.vertices
        return abs(
            (v1[0]-v0[0]) * ((v2[1]-v0[1])*(v3[2]-v0[2]) - (v2[2]-v0[2])*(v3[1]-v0[1])) -
            (v1[1]-v0[1]) * ((v2[0]-v0[0])*(v3[2]-v0[2]) - (v2[2]-v0[2])*(v3[0]-v0[0])) +
            (v1[2]-v0[2]) * ((v2[0]-v0[0])*(v3[1]-v0[1]) - (v2[1]-v0[1])*(v3[0]-v0[0]))
        ) / 6.0


class TENTopologyValidator:
    """
    Tetrahedral Network (TEN) based 3D topology validator
    Builds tetrahedralization of volumetric units and validates:
    - No gaps between adjacent units
    - No overlaps
    - Vertical continuity (floor/ceiling alignment)
    - Boundary alignment with parcel
    - Manifold geometry (watertight)
    """
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance  # meters
        self.tetrahedra: List[Tetrahedron] = []
        self.face_to_tet: dict = {}  # face_key -> [tet_indices]
        self.results: List[ValidationResult] = []
    
    def add_unit(self, suid: str, vertices: List[Tuple[float, float, float]]):
        """Add a volumetric unit by triangulating its boundary"""
        # Simple approach: triangulate each face of the polyhedron
        # In production, use CGAL or pytin for proper 3D constrained Delaunay
        faces = self._extract_faces(vertices)
        for face in faces:
            # Create tetrahedra from face + interior point
            # Simplified: just store face for adjacency detection
            pass
        
        # For now, store as pseudo-tetrahedra for face matching
        self.tetrahedra.append(Tetrahedron(vertices=vertices, suid=suid))
    
    def _extract_faces(self, vertices: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
        """Extract triangular faces from vertex list (simplified)"""
        # This is a placeholder - real implementation uses 3D convex hull
        # or processes POLYHEDRALSURFACE from PostGIS
        faces = []
        n = len(vertices)
        if n >= 4:
            # Simple convex hull approximation
            for i in range(n - 2):
                faces.append([vertices[0], vertices[i+1], vertices[i+2]])
        return faces
    
    def build_face_index(self):
        """Build index of faces to tetrahedra for adjacency detection"""
        self.face_to_tet.clear()
        
        for tet_idx, tet in enumerate(self.tetrahedra):
            for face in tet.get_faces():
                # Normalize face (sort vertices for consistent key)
                face_key = self._normalize_face(face)
                if face_key not in self.face_to_tet:
                    self.face_to_tet[face_key] = []
                self.face_to_tet[face_key].append(tet_idx)
    
    def _normalize_face(self, face: Tuple[Tuple[float, float, float], ...]) -> tuple:
        """Create normalized face key (sorted vertices with tolerance)"""
        def quantize(v):
            return (
                round(v[0] / self.tolerance) * self.tolerance,
                round(v[1] / self.tolerance) * self.tolerance,
                round(v[2] / self.tolerance) * self.tolerance
            )
        return tuple(sorted(quantize(v) for v in face))
    
    def validate(self) -> List[ValidationResult]:
        """Run all topology validations"""
        self.results.clear()
        self.build_face_index()
        
        # 1. Check shared faces (adjacency/overlap)
        self._validate_shared_faces()
        
        # 2. Check vertical continuity
        self._validate_vertical_continuity()
        
        # 3. Check boundary alignment
        self._validate_boundary_alignment()
        
        # 4. Check manifold property
        self._validate_manifold()
        
        return self.results
    
    def _validate_shared_faces(self):
        """Detect shared faces between units"""
        for face_key, tet_indices in self.face_to_tet.items():
            if len(tet_indices) == 2:
                tet1 = self.tetrahedra[tet_indices[0]]
                tet2 = self.tetrahedra[tet_indices[1]]
                
                if tet1.suid != tet2.suid:
                    # Check if face is actually shared (not just touching)
                    if self._faces_match(tet1, tet2, face_key):
                        self.results.append(ValidationResult(
                            suid_1=tet1.suid,
                            suid_2=tet2.suid,
                            relation=TopologyRelation.ADJACENT,
                            details=f"Shared face between {tet1.suid} and {tet2.suid}",
                            severity="info",
                            shared_face_wkt=self._face_to_wkt(face_key)
                        ))
            elif len(tet_indices) > 2:
                # Non-manifold: more than 2 units sharing a face
                for i in range(len(tet_indices)):
                    for j in range(i+1, len(tet_indices)):
                        tet1 = self.tetrahedra[tet_indices[i]]
                        tet2 = self.tetrahedra[tet_indices[j]]
                        if tet1.suid != tet2.suid:
                            self.results.append(ValidationResult(
                                suid_1=tet1.suid,
                                suid_2=tet2.suid,
                                relation=TopologyRelation.OVERLAP,
                                details=f"Non-manifold: {len(tet_indices)} units share face",
                                severity="error",
                                shared_face_wkt=self._face_to_wkt(face_key)
                            ))
    
    def _faces_match(self, tet1: Tetrahedron, tet2: Tetrahedron, face_key: tuple) -> bool:
        """Verify two tetrahedra actually share the same geometric face"""
        # Check if face vertices match within tolerance
        faces1 = {self._normalize_face(f): f for f in tet1.get_faces()}
        faces2 = {self._normalize_face(f): f for f in tet2.get_faces()}
        
        if face_key in faces1 and face_key in faces2:
            f1 = faces1[face_key]
            f2 = faces2[face_key]
            # Check orientation (should be opposite for proper adjacency)
            return self._check_face_orientation(f1, f2)
        return False
    
    def _check_face_orientation(self, face1, face2) -> bool:
        """Check if two faces have opposite orientation (proper adjacency)"""
        # Compute normals
        n1 = self._face_normal(face1)
        n2 = self._face_normal(face2)
        
        # Dot product should be negative for opposite orientation
        dot = sum(a*b for a, b in zip(n1, n2))
        return dot < -0.5  # Allow some tolerance
    
    def _face_normal(self, face) -> Tuple[float, float, float]:
        """Compute face normal (cross product)"""
        v0, v1, v2 = face
        u = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
        v = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
        
        nx = u[1]*v[2] - u[2]*v[1]
        ny = u[2]*v[0] - u[0]*v[2]
        nz = u[0]*v[1] - u[1]*v[0]
        
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            return (nx/length, ny/length, nz/length)
        return (0, 0, 1)
    
    def _validate_vertical_continuity(self):
        """Check floor-ceiling alignment between adjacent floors"""
        # Group by parcel and check vertical stacking
        parcel_units = {}
        for tet in self.tetrahedra:
            parcel_id = self._get_parcel_id(tet.suid)
            if parcel_id not in parcel_units:
                parcel_units[parcel_id] = []
            parcel_units[parcel_id].append(tet)
        
        for parcel_id, units in parcel_units.items():
            # Sort by Z (floor level)
            units.sort(key=lambda t: min(v[2] for v in t.vertices))
            
            for i in range(len(units) - 1):
                upper = units[i]
                lower = units[i+1]
                
                upper_z_min = min(v[2] for v in upper.vertices)
                lower_z_max = max(v[2] for v in lower.vertices)
                
                gap = upper_z_min - lower_z_max
                
                if abs(gap) > self.tolerance:
                    severity = "error" if gap > 0 else "warning"
                    self.results.append(ValidationResult(
                        suid_1=upper.suid,
                        suid_2=lower.suid,
                        relation=TopologyRelation.GAP if gap > 0 else TopologyRelation.OVERLAP,
                        details=f"Vertical {'gap' if gap > 0 else 'overlap'}: {gap:.3f}m",
                        severity=severity
                    ))
    
    def _validate_boundary_alignment(self):
        """Check that unit boundaries align with parcel footprint"""
        # For each parcel, check that child units don't exceed parcel bounds
        parcel_bounds = {}
        unit_bounds = {}
        
        for tet in self.tetrahedra:
            parcel_id = self._get_parcel_id(tet.suid)
            if self._is_parcel(tet.suid):
                parcel_bounds[parcel_id] = self._get_bounds(tet)
            else:
                if parcel_id not in unit_bounds:
                    unit_bounds[parcel_id] = []
                unit_bounds[parcel_id].append((tet.suid, self._get_bounds(tet)))
        
        for parcel_id, p_bounds in parcel_bounds.items():
            if parcel_id in unit_bounds:
                for suid, u_bounds in unit_bounds[parcel_id]:
                    # Check XY bounds
                    if (u_bounds[0] < p_bounds[0] - self.tolerance or
                        u_bounds[1] > p_bounds[1] + self.tolerance or
                        u_bounds[2] < p_bounds[2] - self.tolerance or
                        u_bounds[3] > p_bounds[3] + self.tolerance):
                        self.results.append(ValidationResult(
                            suid_1=parcel_id,
                            suid_2=suid,
                            relation=TopologyRelation.OVERLAP,
                            details=f"Unit {suid} exceeds parcel {parcel_id} footprint",
                            severity="error"
                        ))
    
    def _validate_manifold(self):
        """Check for watertight manifold geometry"""
        for face_key, tet_indices in self.face_to_tet.items():
            if len(tet_indices) == 1:
                # Boundary face - check if it's on parcel boundary
                tet = self.tetrahedra[tet_indices[0]]
                parcel_id = self._get_parcel_id(tet.suid)
                if not self._is_parcel_boundary_face(tet, face_key, parcel_id):
                    self.results.append(ValidationResult(
                        suid_1=tet.suid,
                        suid_2="",
                        relation=TopologyRelation.GAP,
                        details="Unmatched boundary face (potential hole)",
                        severity="warning",
                        shared_face_wkt=self._face_to_wkt(face_key)
                    ))
    
    def _get_parcel_id(self, suid: str) -> str:
        """Extract parcel ID from SUID (simplified)"""
        # In production, query database for parent hierarchy
        return suid.split('-')[0] if '-' in suid else suid[:17] + "00000000"
    
    def _is_parcel(self, suid: str) -> bool:
        return suid.endswith('00000000')
    
    def _is_parcel_boundary_face(self, tet: Tetrahedron, face_key: tuple, parcel_id: str) -> bool:
        """Check if face is on parcel boundary"""
        # Simplified: check if face Z is at parcel min/max Z
        face_z = [v[2] for v in face_key]
        return min(face_z) <= tet.vertices[0][2] + self.tolerance or \
               max(face_z) >= max(v[2] for v in tet.vertices) - self.tolerance
    
    def _get_bounds(self, tet: Tetrahedron) -> Tuple[float, float, float, float]:
        """Get XY bounds (minx, maxx, miny, maxy)"""
        xs = [v[0] for v in tet.vertices]
        ys = [v[1] for v in tet.vertices]
        return (min(xs), max(xs), min(ys), max(ys))
    
    def _face_to_wkt(self, face_key: tuple) -> str:
        """Convert face key to WKT TRIANGLE"""
        coords = ', '.join(f"{v[0]} {v[1]} {v[2]}" for v in face_key)
        return f"TRIANGLE(({coords}, {face_key[0][0]} {face_key[0][1]} {face_key[0][2]}))"


def validate_3d_topology(units: List[dict], tolerance: float = 0.01) -> List[ValidationResult]:
    """
    Convenience function to validate a list of 3D units
    units: [{'suid': str, 'vertices': [(x,y,z), ...]}, ...]
    """
    validator = TENTopologyValidator(tolerance)
    for unit in units:
        validator.add_unit(unit['suid'], unit['vertices'])
    return validator.validate()


# PostGIS-based validation (for production use)
POSTGIS_VALIDATION_SQL = """
-- Validate 3D topology using PostGIS 3D functions
-- Run these as validation queries

-- 1. Check for overlapping volumes
SELECT a.suid as suid_1, b.suid as suid_2, 
       ST_3DIntersection(a.geometry, b.geometry) as intersection
FROM la_spatial_unit a, la_spatial_unit b
WHERE a.suid < b.suid
  AND ST_3DIntersects(a.geometry, b.geometry)
  AND NOT ST_Touches(a.geometry, b.geometry)
  AND a.parent_suid = b.parent_suid;  -- Same parent (e.g., same building)

-- 2. Check for gaps between adjacent floors
SELECT a.suid as upper, b.suid as lower,
       ST_ZMin(a.geometry) - ST_ZMax(b.geometry) as gap
FROM la_spatial_unit a, la_spatial_unit b
WHERE a.parent_suid = b.parent_suid
  AND a.floor_level = b.floor_level + 1
  AND ABS(ST_ZMin(a.geometry) - ST_ZMax(b.geometry)) > 0.01;

-- 3. Check units exceeding parcel bounds
SELECT u.suid, p.suid as parcel
FROM la_spatial_unit u, la_spatial_unit p
WHERE u.parent_suid = p.suid
  AND NOT ST_Within(ST_Force2D(u.geometry), ST_Force2D(p.geometry));

-- 4. Check non-manifold edges (edges shared by >2 faces)
-- Requires topology extension
SELECT * FROM topology.ValidateTopology('cadastral_topo');
"""