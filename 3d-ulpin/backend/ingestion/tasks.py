"""
Celery tasks for data ingestion
"""
from celery import shared_task
from backend.ingestion.celery_app import celery_app
from backend.ingestion.pdal_pipelines import PDALPipeline, create_full_processing_pipeline
from backend.ingestion.floorplan_parser import parse_floor_plan
from backend.ingestion.vertical_delineation import VerticalDelineator, create_volumetric_units_from_building
from backend.ingestion.osm_ingestion import MumbaiOSMIngestion
from backend.core.ulpin import ULPINGenerator
from backend.core.database import get_db_session
from backend.core.models import SpatialUnit, SurveySource
import uuid
import asyncio
import json


@shared_task(bind=True, max_retries=3)
def process_lidar(self, job_id: str, file_path: str, source_id: str):
    """Process LiDAR file with PDAL pipeline"""
    from backend.core.database import get_db_session
    from backend.core.models import SurveySource
    
    async def _process():
        async with get_db_session() as db:
            result = await db.execute(
                "SELECT * FROM cadastral.la_survey_source WHERE source_id = :id",
                {"id": uuid.UUID(source_id)}
            )
            source = result.mappings().first()
            if not source:
                return
            
            source.processing_status = "processing"
            await db.commit()
        
        try:
            pipeline = PDALPipeline()
            pipelines = create_full_processing_pipeline(file_path, "/tmp/output")
            
            for p in pipelines:
                result = pipeline.run_pipeline(p)
                if not result.success:
                    raise Exception(f"PDAL pipeline failed: {result.error}")
            
            # TODO: Extract building footprints and create spatial units
            # This would integrate with ML models
            
            async with get_db_session() as db:
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'completed' WHERE source_id = :id",
                    {"id": uuid.UUID(source_id)}
                )
                await db.commit()
                
        except Exception as e:
            async with get_db_session() as db:
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'failed', metadata = metadata || :meta WHERE source_id = :id",
                    {"id": uuid.UUID(source_id), "meta": json.dumps({"error": str(e)})}
                )
                await db.commit()
            raise
    
    asyncio.run(_process())


@shared_task(bind=True, max_retries=3)
def process_floorplan(self, job_id: str, file_path: str, source_id: str):
    """Process floor plan (DXF/IFC)"""
    from backend.core.database import get_db_session
    from backend.core.models import SurveySource, SpatialUnit
    from backend.core.ulpin import ULPINGenerator
    from backend.ingestion.vertical_delineation import VerticalDelineator
    import uuid
    
    async def _process():
        async with get_db_session() as db:
            result = await db.execute(
                "SELECT * FROM cadastral.la_survey_source WHERE source_id = :id",
                {"id": uuid.UUID(source_id)}
            )
            source = result.mappings().first()
            if not source:
                return
            
            source.processing_status = "processing"
            await db.commit()
        
        try:
            building_plan = parse_floor_plan(file_path)
            
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
            
            async with get_db_session() as db:
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
                
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'completed' WHERE source_id = :id",
                    {"id": uuid.UUID(source_id)}
                )
                await db.commit()
                
        except Exception as e:
            async with get_db_session() as db:
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'failed', metadata = metadata || :meta WHERE source_id = :id",
                    {"id": uuid.UUID(source_id), "meta": json.dumps({"error": str(e)})}
                )
                await db.commit()
            raise
    
    asyncio.run(_process())


@shared_task(bind=True, max_retries=3)
def process_drone(self, job_id: str, file_path: str, source_id: str):
    """Process drone imagery (SfM -> point cloud)"""
    # Placeholder for OpenSfM/Colmap integration
    pass


@shared_task
def run_ml_inference(source_id: str, model_type: str):
    """Run ML inference on processed data"""
    pass


@shared_task(bind=True, max_retries=3)
def process_osm(self, job_id: str, bbox: list, source_id: str, admin_hierarchy: dict = None):
    """Process OSM buildings for given bounding box"""
    from backend.core.database import get_db_session
    from backend.core.models import SurveySource, SpatialUnit
    from backend.core.ulpin import ULPINGenerator
    from backend.ingestion.vertical_delineation import VerticalDelineator
    from backend.ingestion.osm_ingestion import MumbaiOSMIngestion
    import uuid
    
    async def _process():
        async with get_db_session() as db:
            result = await db.execute(
                "SELECT * FROM cadastral.la_survey_source WHERE source_id = :id",
                {"id": uuid.UUID(source_id)}
            )
            source = result.mappings().first()
            if not source:
                return
            
            source.processing_status = "processing"
            await db.commit()
        
        try:
            ingestion = MumbaiOSMIngestion()
            buildings = await ingestion.fetch_bbox(tuple(bbox))
            
            ulpin_gen = ULPINGenerator(admin_hierarchy)
            delineator = VerticalDelineator()
            
            floor_data = ingestion.buildings_to_floor_polygons(buildings)
            units = delineator.extrude_floor_plan(
                f"OSM_{bbox[0]}_{bbox[1]}",
                floor_data,
                ulpin_gen
            )
            
            async with get_db_session() as db:
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
                
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'completed' WHERE source_id = :id",
                    {"id": uuid.UUID(source_id)}
                )
                await db.commit()
                
        except Exception as e:
            async with get_db_session() as db:
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'failed', metadata = metadata || :meta WHERE source_id = :id",
                    {"id": uuid.UUID(source_id), "meta": json.dumps({"error": str(e)})}
                )
                await db.commit()
            raise
    
    asyncio.run(_process())


@shared_task(bind=True, max_retries=3)
def process_mumbai_ward(self, job_id: str, ward: str, source_id: str, admin_hierarchy: dict = None):
    """Process OSM buildings for a Mumbai ward"""
    from backend.core.database import get_db_session
    from backend.core.models import SurveySource, SpatialUnit
    from backend.core.ulpin import ULPINGenerator
    from backend.ingestion.vertical_delineation import VerticalDelineator
    from backend.ingestion.osm_ingestion import MumbaiOSMIngestion
    import uuid
    
    async def _process():
        async with get_db_session() as db:
            result = await db.execute(
                "SELECT * FROM cadastral.la_survey_source WHERE source_id = :id",
                {"id": uuid.UUID(source_id)}
            )
            source = result.mappings().first()
            if not source:
                return
            
            source.processing_status = "processing"
            await db.commit()
        
        try:
            ingestion = MumbaiOSMIngestion()
            buildings = await ingestion.fetch_ward(ward)
            
            ulpin_gen = ULPINGenerator(admin_hierarchy)
            delineator = VerticalDelineator()
            
            floor_data = ingestion.buildings_to_floor_polygons(buildings)
            units = delineator.extrude_floor_plan(
                f"MUMBAI_WARD_{ward}",
                floor_data,
                ulpin_gen
            )
            
            async with get_db_session() as db:
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
                
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'completed' WHERE source_id = :id",
                    {"id": uuid.UUID(source_id)}
                )
                await db.commit()
                
        except Exception as e:
            async with get_db_session() as db:
                await db.execute(
                    "UPDATE cadastral.la_survey_source SET processing_status = 'failed', metadata = metadata || :meta WHERE source_id = :id",
                    {"id": uuid.UUID(source_id), "meta": json.dumps({"error": str(e)})}
                )
                await db.commit()
            raise
    
    asyncio.run(_process())