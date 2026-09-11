"""
Database schema for 3D ULPIN Cadastral System
LADM-aligned with 3D volumetric support
"""
from sqlalchemy import (
    Column, String, Integer, BigInteger, DateTime, Date, 
    ForeignKey, UniqueConstraint, Index, Text, JSON, func
)
from sqlalchemy.dialects.postgresql import UUID, DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import declarative_base, relationship
import uuid

Base = declarative_base()


class Party(Base):
    """LA_Party - Person or organization with rights"""
    __tablename__ = 'la_party'
    
    pid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # person|organization|group
    identifier = Column(String(100))  # national ID, company registration
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Right(Base):
    """LA_Right - Ownership, lease, easement, etc."""
    __tablename__ = 'la_right'
    
    rid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(50), nullable=False)  # ownership|lease|easement|mortgage|restriction
    share_numerator = Column(Integer, default=1)
    share_denominator = Column(Integer, default=1)
    description = Column(Text)
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SpatialUnit(Base):
    """LA_SpatialUnit - 3D volumetric parcel/unit"""
    __tablename__ = 'la_spatial_unit'
    
    suid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uln = Column(String(25), unique=True, nullable=False, index=True)  # 3D ULPIN
    label = Column(String(200))  # "Apt 3B, Tower A"
    type = Column(String(30), nullable=False, index=True)  # parcel|building|floor|unit|utility|airspace|subsurface
    
    # 3D Geometry - POLYHEDRALSURFACE for volumetric solids
    geometry = Column(Text, nullable=False)  # WKT stored as text, PostGIS handles geography
    
    # Elevation bounds (meters, orthometric/MSL)
    z_min = Column(DOUBLE_PRECISION, nullable=False)
    z_max = Column(DOUBLE_PRECISION, nullable=False)
    floor_level = Column(Integer, default=0, index=True)  # -2, -1, 0, 1, 2...
    
    # Hierarchy
    parent_suid = Column(UUID(as_uuid=True), ForeignKey('la_spatial_unit.suid'), index=True)
    
    # Source tracking
    source_id = Column(UUID(as_uuid=True), ForeignKey('la_survey_source.source_id'))
    accuracy_xy = Column(DOUBLE_PRECISION)
    accuracy_z = Column(DOUBLE_PRECISION)
    
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    parent = relationship("SpatialUnit", remote_side=[suid], backref="children")
    source = relationship("SurveySource", backref="spatial_units")
    rrrs = relationship("RRR", back_populates="spatial_unit")


class RRR(Base):
    """LA_RRR - Rights, Restrictions, Responsibilities"""
    __tablename__ = 'la_rrr'
    
    rrr_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    party_ref = Column(UUID(as_uuid=True), ForeignKey('la_party.pid'), nullable=False, index=True)
    right_ref = Column(UUID(as_uuid=True), ForeignKey('la_right.rid'), nullable=False, index=True)
    spatial_unit_ref = Column(UUID(as_uuid=True), ForeignKey('la_spatial_unit.suid'), nullable=False, index=True)
    
    # Time validity
    valid_from = Column(Date, nullable=False)
    valid_until = Column(Date)
    
    # Status
    status = Column(String(20), default='active')  # active|historical|pending
    
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    party = relationship("Party")
    right = relationship("Right")
    spatial_unit = relationship("SpatialUnit", back_populates="rrrs")


class SurveySource(Base):
    """LA_SurveySource - Source data provenance"""
    __tablename__ = 'la_survey_source'
    
    source_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(30), nullable=False, index=True)  # lidar|drone|gnss|floorplan|bim|cadastral
    acquisition_date = Column(Date)
    crs = Column(String(50), nullable=False)  # EPSG:XXXX
    accuracy_xy = Column(DOUBLE_PRECISION)
    accuracy_z = Column(DOUBLE_PRECISION)
    file_path = Column(String(500))  # MinIO/S3 path
    processing_status = Column(String(20), default='pending')  # pending|processing|completed|failed
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TopologyTetra(Base):
    """LA_TopologyTetra - TEN (Tetrahedral Network) for 3D topology"""
    __tablename__ = 'la_topology_tetra'
    
    tet_id = Column(BigInteger, primary_key=True, autoincrement=True)
    suid_1 = Column(UUID(as_uuid=True), ForeignKey('la_spatial_unit.suid'), nullable=False, index=True)
    suid_2 = Column(UUID(as_uuid=True), ForeignKey('la_spatial_unit.suid'), nullable=False, index=True)
    shared_face = Column(Text)  # WKT TRIANGLE
    relationship = Column(String(20), nullable=False)  # adjacent|overlap|gap|contain|touch
    meta = Column('metadata', JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Ensure unique pair
    __table_args__ = (
        Index('ix_topo_pair', 'suid_1', 'suid_2', unique=True),
    )


class ULPINSequence(Base):
    """ULPIN component sequences for deterministic generation"""
    __tablename__ = 'ulpin_sequence'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    component = Column(String(20), unique=True, nullable=False)  # state|district|subdist|village|parcel|vertical|unit
    current_value = Column(BigInteger, default=0)
    max_value = Column(BigInteger, nullable=False)


# SQL for PostGIS extensions and indexes (run as raw SQL)
POSTGIS_SETUP_SQL = """
-- Enable PostGIS and 3D extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- 3D spatial index on geometry (using geography for 3D)
CREATE INDEX IF NOT EXISTS idx_spatial_unit_geom_3d 
ON la_spatial_unit USING GIST (ST_GeomFromText(geometry, 4326));

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_spatial_unit_parent_type 
ON la_spatial_unit (parent_suid, type);

CREATE INDEX IF NOT EXISTS idx_spatial_unit_z_bounds 
ON la_spatial_unit (z_min, z_max);

CREATE INDEX IF NOT EXISTS idx_rrr_spatial_unit_valid 
ON la_rrr (spatial_unit_ref, valid_from, valid_until) 
WHERE status = 'active';

-- Function to generate 3D ULPIN
CREATE OR REPLACE FUNCTION generate_3d_ulpin(
    p_state CHAR(2),
    p_district CHAR(3),
    p_subdist CHAR(2),
    p_village CHAR(4),
    p_parcel CHAR(6),
    p_vertical CHAR(4),
    p_unit CHAR(4)
) RETURNS VARCHAR(25) AS $$
BEGIN
    RETURN p_state || p_district || p_subdist || p_village || p_parcel || p_vertical || p_unit;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to parse 3D ULPIN components
CREATE OR REPLACE FUNCTION parse_3d_ulpin(p_ulpin VARCHAR(25)) 
RETURNS TABLE (
    state CHAR(2), district CHAR(3), subdist CHAR(2), 
    village CHAR(4), parcel CHAR(6), vertical CHAR(4), unit CHAR(4)
) AS $$
BEGIN
    IF LENGTH(p_ulpin) <> 25 THEN
        RAISE EXCEPTION 'Invalid 3D ULPIN length: %', LENGTH(p_ulpin);
    END IF;
    RETURN QUERY SELECT 
        SUBSTRING(p_ulpin FROM 1 FOR 2),
        SUBSTRING(p_ulpin FROM 3 FOR 3),
        SUBSTRING(p_ulpin FROM 6 FOR 2),
        SUBSTRING(p_ulpin FROM 8 FOR 4),
        SUBSTRING(p_ulpin FROM 12 FOR 6),
        SUBSTRING(p_ulpin FROM 18 FOR 4),
        SUBSTRING(p_ulpin FROM 22 FOR 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to compute floor_level from Z (assuming ~3m per floor)
CREATE OR REPLACE FUNCTION compute_floor_level(z_base DOUBLE PRECISION, z_top DOUBLE PRECISION, ground_z DOUBLE PRECISION DEFAULT 0)
RETURNS INTEGER AS $$
DECLARE
    floor_height CONSTANT DOUBLE PRECISION := 3.0;
    mid_z DOUBLE PRECISION := (z_base + z_top) / 2.0;
BEGIN
    RETURN ROUND((mid_z - ground_z) / floor_height)::INTEGER;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


def get_create_table_sql():
    """Generate CREATE TABLE statements for all models"""
    from sqlalchemy.schema import CreateTable
    from sqlalchemy import MetaData
    
    metadata = MetaData()
    for table in Base.metadata.tables.values():
        table.metadata = metadata
    
    statements = []
    for table in Base.metadata.sorted_tables:
        statements.append(str(CreateTable(table).compile(dialect=None)))
    
    return ';\n\n'.join(statements) + ';'