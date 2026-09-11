"""
FastAPI Application - 3D ULPIN Cadastral System
OGC API - Features compliant
"""
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from backend.core.database import init_database, close_database, get_db, get_sync_session, get_db_session
from backend.core.models import (
    SpatialUnit, Party, Right, RRR, SurveySource, 
    TopologyTetra, Base, POSTGIS_SETUP_SQL
)
from backend.core.ulpin import generate_3d_ulpin, ULPINGenerator, ULPINComponents
from backend.core.topology import validate_3d_topology, ValidationResult, TopologyRelation
from backend.ingestion.pdal_pipelines import PDALPipeline, create_full_processing_pipeline
from backend.ingestion.floorplan_parser import parse_floor_plan, FloorPolygon, BuildingFloorPlan
from backend.ingestion.vertical_delineation import (
    VerticalDelineator, VolumetricUnit, create_volumetric_units_from_building
)
from backend.ingestion.osm_ingestion import MumbaiOSMIngestion, MumbaiBounds
from backend.ml.inference import (
    FloorSegmenterONNX, BuildingExtractorONNX, VerticalDelineatorML,
    load_models, run_ml_inference, InferenceResult
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlalchemy.orm import selectinload
import uuid


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    print("3D ULPIN System started")
    yield
    # Shutdown
    await close_database()
    print("3D ULPIN System stopped")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="3D ULPIN Cadastral System",
    description="OGC API - Features compliant 3D cadastral management",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SpatialUnitCreate(BaseModel):
    label: str
    type: str
    geometry_wkt: str
    z_min: float
    z_max: float
    floor_level: int = 0
    parent_ulpin: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SpatialUnitResponse(BaseModel):
    suid: str
    uln: str
    label: str
    type: str
    geometry: str
    z_min: float
    z_max: float
    floor_level: int
    parent_ulpin: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    
    class Config:
        from_attributes = True


class ULPINGenerateRequest(BaseModel):
    x: float
    y: float
    z: float = 0
    level: str = "unit"  # parcel|building|floor|unit
    parent_ulpin: Optional[str] = None
    floor_level: int = 0
    unit_index: int = 1
    admin: Optional[Dict[str, str]] = None


class ULPINResponse(BaseModel):
    ulpin: str
    components: Dict[str, str]


class TopologyValidationRequest(BaseModel):
    units: List[Dict]  # [{'suid': str, 'vertices': [[x,y,z], ...]}, ...]
    tolerance: float = 0.01


class TopologyValidationResponse(BaseModel):
    results: List[Dict]
    summary: Dict[str, int]


class IngestionJobRequest(BaseModel):
    source_type: str  # lidar|drone|floorplan|gnss|bim
    file_path: str
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = {}


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class BuildingExtrudeRequest(BaseModel):
    building_id: str
    footprint_wkt: str
    num_floors: int
    floor_height: float = 3.0
    ground_z: float = 0.0
    unit_layout: str = "grid"  # grid|perimeter


class OSMIngestionRequest(BaseModel):
    """OSM/Overpass API ingestion request"""
    bbox: List[float]  # [west, south, east, north]
    admin: Optional[Dict[str, str]] = None  # state, district, subdistrict, village


class MumbaiWardRequest(BaseModel):
    """Mumbai ward ingestion request"""
    ward: str  # A, B, C, D, E, F_N, F_S, G_N, G_S, H_E, H_W, K_E, K_W, L, M_E, M_W, N, P_N, P_S, R_C, R_N, R_S, S, T
    admin: Optional[Dict[str, str]] = None


# ============================================================================
# HEALTH & ROOT
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "3D ULPIN Cadastral System",
        "version": "1.0.0",
        "standards": ["OGC API - Features", "ISO 19152 LADM", "CityGML 3.0"],
        "endpoints": {
            "collections": "/collections",
            "ulpin": "/ulpin",
            "topology": "/topology",
            "ingestion": "/ingestion",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# OGC API - FEATURES: COLLECTIONS
# ============================================================================

@app.get("/collections")
async def get_collections():
    """OGC API - Features: Collections"""
    return {
        "collections": [
            {
                "id": "spatial_units",
                "title": "3D Spatial Units",
                "description": "Volumetric cadastral parcels, buildings, floors, units",
                "links": [
                    {"rel": "items", "href": "/collections/spatial_units/items", "type": "application/geo+json"},
                    {"rel": "self", "href": "/collections/spatial_units", "type": "application/json"}
                ],
                "extent": {
                    "spatial": {"bbox": [[-180, -90, 180, 90]]},
                    "temporal": {"interval": [["2024-01-01T00:00:00Z", None]]}
                },
                "crs": ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
                "itemType": "feature"
            },
            {
                "id": "parties",
                "title": "Parties (Owners)",
                "description": "Legal persons and organizations",
                "links": [
                    {"rel": "items", "href": "/collections/parties/items", "type": "application/geo+json"}
                ]
            },
            {
                "id": "rights",
                "title": "Rights",
                "description": "Ownership, lease, easement rights",
                "links": [
                    {"rel": "items", "href": "/collections/rights/items", "type": "application/geo+json"}
                ]
            },
            {
                "id": "survey_sources",
                "title": "Survey Sources",
                "description": "LiDAR, drone, GNSS, floor plan sources",
                "links": [
                    {"rel": "items", "href": "/collections/survey_sources/items", "type": "application/geo+json"}
                ]
            }
        ],
        "links": [{"rel": "self", "href": "/collections", "type": "application/json"}]
    }


@app.get("/collections/{collection_id}/items")
async def get_collection_items(
    collection_id: str,
    bbox: Optional[str] = Query(None, description="bbox=minx,miny,maxx,maxy"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    uln: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    floor_level: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """OGC API - Features: Feature Collection"""
    
    if collection_id == "spatial_units":
        query = select(SpatialUnit)
        
        if uln:
            query = query.where(SpatialUnit.uln == uln)
        if type:
            query = query.where(SpatialUnit.type == type)
        if floor_level is not None:
            query = query.where(SpatialUnit.floor_level == floor_level)
        
        # BBox filter (simplified - use PostGIS in production)
        if bbox:
            try:
                minx, miny, maxx, maxy = map(float, bbox.split(','))
                query = query.where(text(
                    "ST_Intersects(ST_GeomFromText(geometry, 4326), ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))"
                )).params(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            except:
                pass
        
        total = await db.scalar(select(func.count()).select_from(query.subquery()))
        query = query.limit(limit).offset(offset)
        result = await db.execute(query)
        units = result.scalars().all()
        
        features = []
        for unit in units:
            features.append({
                "type": "Feature",
                "id": str(unit.suid),
                "geometry": json.loads(unit.geometry) if isinstance(unit.geometry, str) else unit.geometry,
                "properties": {
                    "uln": unit.uln,
                    "label": unit.label,
                    "type": unit.type,
                    "z_min": unit.z_min,
                    "z_max": unit.z_max,
                    "floor_level": unit.floor_level,
                    "parent_ulpin": str(unit.parent_suid) if unit.parent_suid else None,
                    "metadata": unit.metadata
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "numberMatched": total,
            "numberReturned": len(features),
            "links": [
                {"rel": "self", "href": f"/collections/{collection_id}/items?limit={limit}&offset={offset}"},
                {"rel": "next", "href": f"/collections/{collection_id}/items?limit={limit}&offset={offset+limit}"} if offset + limit < total else None
            ]
        }
    
    elif collection_id == "parties":
        query = select(Party).limit(limit).offset(offset)
        result = await db.execute(query)
        parties = result.scalars().all()
        
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": str(p.pid),
                "geometry": None,
                "properties": {"pid": str(p.pid), "name": p.name, "type": p.type}
            } for p in parties]
        }
    
    elif collection_id == "survey_sources":
        query = select(SurveySource).limit(limit).offset(offset)
        result = await db.execute(query)
        sources = result.scalars().all()
        
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": str(s.source_id),
                "geometry": None,
                "properties": {
                    "source_id": str(s.source_id),
                    "type": s.type,
                    "acquisition_date": s.acquisition_date.isoformat() if s.acquisition_date else None,
                    "crs": s.crs,
                    "status": s.processing_status
                }
            } for s in sources]
        }
    
    raise HTTPException(404, f"Collection {collection_id} not found")


@app.get("/collections/spatial_units/items/{feature_id}")
async def get_spatial_unit(feature_id: str, db: AsyncSession = Depends(get_db)):
    """Get single spatial unit by SUID"""
    result = await db.execute(select(SpatialUnit).where(SpatialUnit.suid == feature_id))
    unit = result.scalar_one_or_none()
    
    if not unit:
        raise HTTPException(404, "Spatial unit not found")
    
    return {
        "type": "Feature",
        "id": str(unit.suid),
        "geometry": json.loads(unit.geometry) if isinstance(unit.geometry, str) else unit.geometry,
        "properties": {
            "uln": unit.uln,
            "label": unit.label,
            "type": unit.type,
            "z_min": unit.z_min,
            "z_max": unit.z_max,
            "floor_level": unit.floor_level,
            "parent_ulpin": str(unit.parent_suid) if unit.parent_suid else None,
            "metadata": unit.metadata,
            "created_at": unit.created_at.isoformat() if unit.created_at else None
        }
    }


# ============================================================================
# 3D ULPIN ENDPOINTS
# ============================================================================

@app.post("/ulpin/generate", response_model=ULPINResponse)
async def generate_ulpin(request: ULPINGenerateRequest):
    """Generate 3D ULPIN from coordinates"""
    ulpin = generate_3d_ulpin(
        x=request.x,
        y=request.y,
        z=request.z,
        level=request.level,
        parent_ulpin=request.parent_ulpin,
        floor_level=request.floor_level,
        unit_index=request.unit_index,
        admin=request.admin
    )
    
    comps = ULPINComponents.from_string(ulpin)
    return ULPINResponse(
        ulpin=ulpin,
        components={
            "state": comps.state,
            "district": comps.district,
            "subdistrict": comps.subdistrict,
            "village": comps.village,
            "parcel": comps.parcel,
            "vertical": comps.vertical,
            "unit": comps.unit
        }
    )


@app.post("/ulpin/parse")
async def parse_ulpin(ulpin: str):
    """Parse 3D ULPIN into components"""
    comps = ULPINComponents.from_string(ulpin)
    return {
        "ulpin": ulpin,
        "components": {
            "state": comps.state,
            "district": comps.district,
            "subdistrict": comps.subdistrict,
            "village": comps.village,
            "parcel": comps.parcel,
            "vertical": comps.vertical,
            "unit": comps.unit
        },
        "hierarchy": comps.get_hierarchy_path()
    }


@app.get("/ulpin/hierarchy/{ulpin}")
async def get_ulpin_hierarchy(ulpin: str, db: AsyncSession = Depends(get_db)):
    """Get full hierarchy for a ULPIN"""
    comps = ULPINComponents.from_string(ulpin)
    hierarchy = comps.get_hierarchy_path()
    
    results = []
    for h in hierarchy:
        result = await db.execute(select(SpatialUnit).where(SpatialUnit.uln == h))
        unit = result.scalar_one_or_none()
        if unit:
            results.append({
                "uln": unit.uln,
                "label": unit.label,
                "type": unit.type,
                "floor_level": unit.floor_level
            })
    
    return {"ulpin": ulpin, "hierarchy": results}


# ============================================================================
# TOPOLOGY VALIDATION
# ============================================================================

@app.post("/topology/validate", response_model=TopologyValidationResponse)
async def validate_topology(request: TopologyValidationRequest):
    """Validate 3D topology using TEN model"""
    results = validate_3d_topology(request.units, request.tolerance)
    
    summary = {"error": 0, "warning": 0, "info": 0}
    for r in results:
        summary[r.severity] = summary.get(r.severity, 0) + 1
    
    return TopologyValidationResponse(
        results=[{
            "suid_1": r.suid_1,
            "suid_2": r.suid_2,
            "relation": r.relation.value,
            "details": r.details,
            "severity": r.severity,
            "shared_face_wkt": r.shared_face_wkt
        } for r in results],
        summary=summary
    )


@app.post("/topology/validate/database")
async def validate_database_topology(
    parcel_ulpin: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Validate topology for units in database"""
    query = select(SpatialUnit)
    if parcel_ulpin:
        # Get all units under this parcel
        parcel = await db.execute(select(SpatialUnit).where(SpatialUnit.uln == parcel_ulpin))
        parcel_obj = parcel.scalar_one_or_none()
        if parcel_obj:
            query = query.where(SpatialUnit.parent_suid == parcel_obj.suid)
    
    result = await db.execute(query.limit(1000))
    units = result.scalars().all()
    
    # Convert to validator format
    unit_data = []
    for unit in units:
        # Extract vertices from geometry WKT (simplified)
        unit_data.append({
            'suid': str(unit.suid),
            'vertices': []  # Would parse from geometry
        })
    
    results = validate_3d_topology(unit_data)
    
    return TopologyValidationResponse(
        results=[{
            "suid_1": r.suid_1,
            "suid_2": r.suid_2,
            "relation": r.relation.value,
            "details": r.details,
            "severity": r.severity
        } for r in results],
        summary={"total_units": len(units), "issues": len(results)}
    )


# ============================================================================
# INGESTION ENDPOINTS
# ============================================================================

@app.post("/ingestion/submit", response_model=IngestionJobResponse)
async def submit_ingestion_job(request: IngestionJobRequest, background_tasks: BackgroundTasks):
    """Submit data ingestion job"""
    job_id = str(uuid.uuid4())
    
    # Create source record
    async with get_db_session() as db:
        source = SurveySource(
            source_id=uuid.UUID(job_id),
            type=request.source_type,
            crs=request.crs,
            file_path=request.file_path,
            processing_status="pending",
            metadata=request.metadata
        )
        db.add(source)
        await db.commit()
    
    # Process in background
    background_tasks.add_task(process_ingestion_job, job_id, request)
    
    return IngestionJobResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion job submitted"
    )


async def process_ingestion_job(job_id: str, request: IngestionJobRequest):
    """Background task for ingestion processing"""
    async with get_db_session() as db:
        # Update status
        result = await db.execute(select(SurveySource).where(SurveySource.source_id == job_id))
        source = result.scalar_one_or_none()
        if not source:
            return
        
        source.processing_status = "processing"
        await db.commit()
        
        try:
            if request.source_type == "lidar":
                await process_lidar(job_id, request.file_path, db)
            elif request.source_type == "floorplan":
                await process_floorplan(job_id, request.file_path, db)
            elif request.source_type == "drone":
                await process_drone(job_id, request.file_path, db)
            
            source.processing_status = "completed"
        except Exception as e:
            source.processing_status = "failed"
            source.metadata["error"] = str(e)
        
        await db.commit()


async def process_lidar(job_id: str, file_path: str, db: AsyncSession):
    """Process LiDAR file"""
    pipeline = PDALPipeline()
    
    # Run full processing chain
    pipelines = create_full_processing_pipeline(file_path, "/tmp/output")
    
    for p in pipelines:
        result = pipeline.run_pipeline(p)
        if not result.success:
            raise Exception(f"Pipeline failed: {result.error}")
    
    # Extract building footprints and create spatial units
    # This would connect to the vertical delineation
    pass


async def process_floorplan(job_id: str, file_path: str, db: AsyncSession):
    """Process floor plan (DXF/IFC)"""
    building_plan = parse_floor_plan(file_path)
    
    # Generate volumetric units
    ulpin_gen = ULPINGenerator()
    delineator = VerticalDelineator()
    
    floor_data = []
    for fp in building_plan.floors:
        floor_data.append({
            'level': fp.level,
            'z_base': fp.z_base,
            'height': fp.height,
            'polygon_wkt': fp.polygon_wkt,
            'unit_id': fp.unit_id,
            'unit_type': fp.unit_type
        })
    
    units = delineator.extrude_floor_plan(
        building_plan.building_id,
        floor_data,
        ulpin_gen
    )
    
    # Save to database
    for unit in units:
        su = SpatialUnit(
            suid=uuid.UUID(unit.suid) if '-' in unit.suid else uuid.uuid4(),
            uln=unit.uln,
            label=unit.label,
            type=unit.unit_type,
            geometry=unit.geometry_wkt,
            z_min=unit.z_min,
            z_max=unit.z_max,
            floor_level=unit.floor_level,
            parent_suid=uuid.UUID(unit.parent_suid) if unit.parent_suid else None,
            metadata=unit.metadata
        )
        db.add(su)
    
    await db.commit()


async def process_drone(job_id: str, file_path: str, db: AsyncSession):
    """Process drone imagery (SfM -> point cloud)"""
    # Would use OpenSfM/Colmap
    pass


@app.get("/ingestion/jobs/{job_id}")
async def get_ingestion_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get ingestion job status"""
    result = await db.execute(select(SurveySource).where(SurveySource.source_id == job_id))
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(404, "Job not found")
    
    return {
        "job_id": str(source.source_id),
        "type": source.type,
        "status": source.processing_status,
        "file_path": source.file_path,
        "metadata": source.metadata,
        "created_at": source.created_at.isoformat() if source.created_at else None
    }


# ============================================================================
# BUILDING EXTRUSION
# ============================================================================

@app.post("/buildings/extrude")
async def extrude_building(request: BuildingExtrudeRequest):
    """Create 3D volumetric units from building footprint"""
    from shapely.wkt import loads
    
    footprint = loads(request.footprint_wkt)
    
    units = create_volumetric_units_from_building(
        building_id=request.building_id,
        footprint=footprint,
        num_floors=request.num_floors,
        floor_height=request.floor_height,
        ground_z=request.ground_z
    )
    
    return {
        "building_id": request.building_id,
        "units_created": len(units),
        "units": [{
            "suid": u.suid,
            "uln": u.uln,
            "label": u.label,
            "type": u.unit_type,
            "z_min": u.z_min,
            "z_max": u.z_max,
            "floor_level": u.floor_level
        } for u in units]
    }


# ============================================================================
# OSM INGESTION ENDPOINTS
# ============================================================================

@app.post("/ingestion/osm")
async def ingest_osm_buildings(request: OSMIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest buildings from OSM/Overpass API for a bounding box"""
    job_id = str(uuid.uuid4())
    
    # Validate bbox
    if len(request.bbox) != 4:
        raise HTTPException(400, "bbox must be [west, south, east, north]")
    
    async with get_db_session() as db:
        source = SurveySource(
            source_id=uuid.UUID(job_id),
            type="osm",
            crs="EPSG:4326",
            file_path=f"OSM:{request.bbox}",
            processing_status="pending",
            metadata={
                "bbox": request.bbox,
                "source": "overpass",
                "admin": request.admin or {}
            }
        )
        db.add(source)
        await db.commit()
    
    # Queue background task
    from backend.ingestion.tasks import process_osm
    process_osm.delay(job_id, request.bbox, job_id, request.admin)
    
    return IngestionJobResponse(
        job_id=job_id,
        status="pending",
        message="OSM ingestion job submitted"
    )


@app.post("/ingestion/mumbai-ward")
async def ingest_mumbai_ward(request: MumbaiWardRequest, background_tasks: BackgroundTasks):
    """Ingest buildings for a specific Mumbai ward"""
    job_id = str(uuid.uuid4())
    
    valid_wards = set(MumbaiBounds.WARDS.keys())
    # Check ward validity (case-insensitive)
    if request.ward.upper() not in [w.upper() for w in valid_wards]:
        raise HTTPException(400, f"Invalid ward. Valid wards: {list(valid_wards)}")
    
    async with get_db_session() as db:
        source = SurveySource(
            source_id=uuid.UUID(job_id),
            type="osm",
            crs="EPSG:4326",
            file_path=f"OSM:MUMBAI_WARD_{request.ward.upper()}",
            processing_status="pending",
            metadata={
                "ward": request.ward.upper(),
                "source": "overpass",
                "admin": request.admin or {}
            }
        )
        db.add(source)
        await db.commit()
    
    # Queue background task
    from backend.ingestion.tasks import process_mumbai_ward
    process_mumbai_ward.delay(job_id, request.ward.upper(), job_id, request.admin)
    
    return IngestionJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Mumbai ward {request.ward.upper()} ingestion job submitted"
    )


@app.get("/ingestion/mumbai-wards")
async def list_mumbai_wards():
    """List all Mumbai wards with bounding boxes"""
    return {
        "wards": {
            ward: {
                "bbox": bbox,
                "name": ward
            } for ward, bbox in MumbaiBounds.WARDS.items()
        }
    }


@app.post("/ingestion/osm/preview")
async def preview_osm_buildings(request: OSMIngestionRequest):
    """Preview buildings from OSM without saving to database"""
    from backend.ingestion.osm_ingestion import MumbaiOSMIngestion
    
    ingestion = MumbaiOSMIngestion()
    buildings = await ingestion.fetch_bbox(tuple(request.bbox))
    
    # Convert to summary
    summary = {
        "total_buildings": len(buildings),
        "by_type": {},
        "by_levels": {},
        "bbox": request.bbox
    }
    
    for b in buildings:
        btype = b.building_type
        summary["by_type"][btype] = summary["by_type"].get(btype, 0) + 1
        
        levels = b.levels or 0
        summary["by_levels"][str(levels)] = summary["by_levels"].get(str(levels), 0) + 1
    
    # Return first 10 as sample
    sample = [{
        "osm_id": b.osm_id,
        "name": b.name,
        "building_type": b.building_type,
        "levels": b.levels,
        "height": b.height,
        "centroid": b.centroid,
        "area": b.geometry.area
    } for b in buildings[:10]]
    
    return {"summary": summary, "sample": sample}


# ============================================================================
# ML INFERENCE ENDPOINTS
# ============================================================================

# Global model instances (loaded on startup)
_floor_segmenter: FloorSegmenterONNX = None
_building_extractor: BuildingExtractorONNX = None
_vertical_delineator_ml: VerticalDelineatorML = None


@app.on_event("startup")
async def load_ml_models():
    """Load ML models on startup"""
    global _floor_segmenter, _building_extractor, _vertical_delineator_ml
    
    model_dir = Path("ml/checkpoints")
    fs_path = model_dir / "floor_segmenter.onnx"
    be_path = model_dir / "building_extractor.onnx"
    
    if fs_path.exists():
        _floor_segmenter = FloorSegmenterONNX(str(fs_path))
        print("Loaded floor segmenter model")
    
    if be_path.exists():
        _building_extractor = BuildingExtractorONNX(str(be_path))
        print("Loaded building extractor model")
    
    if _floor_segmenter:
        _vertical_delineator_ml = VerticalDelineatorML(_floor_segmenter, _building_extractor)


class FloorSegmentationRequest(BaseModel):
    points: List[List[float]]  # [[x, y, z, r, g, b], ...]


class FloorSegmentationResponse(BaseModel):
    success: bool
    predictions: Optional[List[int]] = None
    probabilities: Optional[List[List[float]]] = None
    metadata: Dict = {}
    error: Optional[str] = None


@app.post("/ml/floor-segmentation", response_model=FloorSegmentationResponse)
async def floor_segmentation(request: FloorSegmentationRequest):
    """Run floor segmentation on point cloud"""
    if not _floor_segmenter:
        raise HTTPException(503, "Floor segmentation model not loaded")
    
    points = np.array(request.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise HTTPException(400, "Points must be Nx6 array [x, y, z, r, g, b]")
    
    result = _floor_segmenter.predict(points)
    
    return FloorSegmentationResponse(
        success=result.success,
        predictions=result.predictions.tolist() if result.predictions is not None else None,
        probabilities=result.probabilities.tolist() if result.probabilities is not None else None,
        metadata=result.metadata or {},
        error=result.error
    )


class BuildingExtractionRequest(BaseModel):
    image: List[List[List[float]]]  # [C, H, W]
    patch_size: int = 256
    overlap: int = 32


class BuildingExtractionResponse(BaseModel):
    success: bool
    predictions: Optional[List[List[int]]] = None
    probabilities: Optional[List[List[List[float]]]] = None
    metadata: Dict = {}
    error: Optional[str] = None


@app.post("/ml/building-extraction", response_model=BuildingExtractionResponse)
async def building_extraction(request: BuildingExtractionRequest):
    """Run building footprint extraction on DSM/Ortho"""
    if not _building_extractor:
        raise HTTPException(503, "Building extraction model not loaded")
    
    image = np.array(request.image, dtype=np.float32)
    if image.ndim != 3:
        raise HTTPException(400, "Image must be 3D array [C, H, W]")
    
    result = _building_extractor.predict(image, request.patch_size, request.overlap)
    
    return BuildingExtractionResponse(
        success=result.success,
        predictions=result.predictions.tolist() if result.predictions is not None else None,
        probabilities=result.probabilities.tolist() if result.probabilities is not None else None,
        metadata=result.metadata or {},
        error=result.error
    )


class VerticalDelineationRequest(BaseModel):
    points: List[List[float]]  # [[x, y, z, r, g, b], ...]
    building_footprint: Optional[List[List[float]]] = None


class VerticalDelineationResponse(BaseModel):
    success: bool
    floor_levels: Optional[List[Dict]] = None
    units: Optional[List[Dict]] = None
    segmentation: Optional[Dict] = None
    error: Optional[str] = None


@app.post("/ml/vertical-delineation", response_model=VerticalDelineationResponse)
async def vertical_delineation(request: VerticalDelineationRequest):
    """Run full vertical delineation pipeline"""
    if not _vertical_delineator_ml:
        raise HTTPException(503, "Vertical delineation models not loaded")
    
    points = np.array(request.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise HTTPException(400, "Points must be Nx6 array [x, y, z, r, g, b]")
    
    footprint = None
    if request.building_footprint:
        footprint = np.array(request.building_footprint, dtype=np.float32)
    
    result = _vertical_delineator_ml.delineate_from_pointcloud(points, footprint)
    
    return VerticalDelineationResponse(**result)


@app.get("/ml/models/status")
async def ml_models_status():
    """Check ML model loading status"""
    return {
        "floor_segmenter": _floor_segmenter is not None,
        "building_extractor": _building_extractor is not None,
        "vertical_delineator": _vertical_delineator_ml is not None,
        "model_dir": "ml/checkpoints"
    }


# ============================================================================
# 3D TILES EXPORT
# ============================================================================

@app.get("/tiles/{collection_id}/tileset.json")
async def get_tileset(collection_id: str, db: AsyncSession = Depends(get_db)):
    """Generate 3D Tiles tileset.json for CesiumJS"""
    # Simplified tileset - in production use py3dtiles
    return {
        "asset": {"version": "1.1", "generator": "3D ULPIN System"},
        "geometricError": 1000,
        "root": {
            "boundingVolume": {
                "box": [0, 0, 0, 1000, 0, 0, 0, 1000, 0, 0, 0, 500]
            },
            "geometricError": 500,
            "refine": "REPLACE",
            "content": {
                "url": f"/tiles/{collection_id}/root.b3dm"
            },
            "children": []
        }
    }


@app.get("/tiles/{collection_id}/{tile_file}")
async def get_tile(collection_id: str, tile_file: str):
    """Serve 3D tile files (b3dm, gltf, etc.)"""
    # In production, serve from MinIO/S3 or generated cache
    raise HTTPException(404, "Tile not found - generate tiles first")


# ============================================================================
# ADMIN / MANAGEMENT
# ============================================================================

@app.post("/admin/spatial-units")
async def create_spatial_unit(unit: SpatialUnitCreate, db: AsyncSession = Depends(get_db)):
    """Create spatial unit manually"""
    suid = uuid.uuid4()
    ulpin = f"MANUAL{suid.int % 10**18:018d}"  # Placeholder
    
    su = SpatialUnit(
        suid=suid,
        uln=ulpin,
        label=unit.label,
        type=unit.type,
        geometry=unit.geometry_wkt,
        z_min=unit.z_min,
        z_max=unit.z_max,
        floor_level=unit.floor_level,
        metadata=unit.metadata
    )
    
    if unit.parent_ulpin:
        result = await db.execute(select(SpatialUnit).where(SpatialUnit.uln == unit.parent_ulpin))
        parent = result.scalar_one_or_none()
        if parent:
            su.parent_suid = parent.suid
    
    db.add(su)
    await db.commit()
    await db.refresh(su)
    
    return SpatialUnitResponse.model_validate(su)


@app.get("/admin/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """System statistics"""
    total_units = await db.scalar(select(func.count(SpatialUnit.suid)))
    by_type = await db.execute(
        select(SpatialUnit.type, func.count(SpatialUnit.suid))
        .group_by(SpatialUnit.type)
    )
    by_level = await db.execute(
        select(SpatialUnit.floor_level, func.count(SpatialUnit.suid))
        .group_by(SpatialUnit.floor_level)
    )
    sources = await db.scalar(select(func.count(SurveySource.source_id)))
    
    return {
        "total_spatial_units": total_units,
        "by_type": dict(by_type.all()),
        "by_floor_level": dict(by_level.all()),
        "survey_sources": sources
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)