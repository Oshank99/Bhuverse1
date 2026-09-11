"""Initial 3D ULPIN Cadastral Schema

Revision ID: 001
Revises: 
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, DOUBLE_PRECISION
import uuid

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable PostGIS extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis_raster;')
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis_topology;')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    
    # la_party
    op.create_table(
        'la_party',
        sa.Column('pid', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('identifier', sa.String(100)),
        sa.Column('metadata', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        schema='cadastral'
    )
    
    # la_right
    op.create_table(
        'la_right',
        sa.Column('rid', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('share_numerator', sa.Integer, default=1),
        sa.Column('share_denominator', sa.Integer, default=1),
        sa.Column('description', sa.Text),
        sa.Column('metadata', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        schema='cadastral'
    )
    
    # la_survey_source (must be first due to FK from la_spatial_unit)
    op.create_table(
        'la_survey_source',
        sa.Column('source_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('type', sa.String(30), nullable=False, index=True),
        sa.Column('acquisition_date', sa.Date),
        sa.Column('crs', sa.String(50), nullable=False),
        sa.Column('accuracy_xy', DOUBLE_PRECISION),
        sa.Column('accuracy_z', DOUBLE_PRECISION),
        sa.Column('file_path', sa.String(500)),
        sa.Column('processing_status', sa.String(20), default='pending'),
        sa.Column('meta', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        schema='cadastral'
    )
    
    # la_spatial_unit
    op.create_table(
        'la_spatial_unit',
        sa.Column('suid', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('uln', sa.String(25), unique=True, nullable=False, index=True),
        sa.Column('label', sa.String(200)),
        sa.Column('type', sa.String(30), nullable=False, index=True),
        sa.Column('geometry', sa.Text, nullable=False),  # WKT for POLYHEDRALSURFACE Z
        sa.Column('z_min', DOUBLE_PRECISION, nullable=False),
        sa.Column('z_max', DOUBLE_PRECISION, nullable=False),
        sa.Column('floor_level', sa.Integer, default=0, index=True),
        sa.Column('parent_suid', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_spatial_unit.suid'), index=True),
        sa.Column('source_id', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_survey_source.source_id')),
        sa.Column('accuracy_xy', DOUBLE_PRECISION),
        sa.Column('accuracy_z', DOUBLE_PRECISION),
        sa.Column('meta', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        schema='cadastral'
    )
    
    # la_rrr
    op.create_table(
        'la_rrr',
        sa.Column('rrr_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('party_ref', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_party.pid'), nullable=False, index=True),
        sa.Column('right_ref', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_right.rid'), nullable=False, index=True),
        sa.Column('spatial_unit_ref', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_spatial_unit.suid'), nullable=False, index=True),
        sa.Column('valid_from', sa.Date, nullable=False),
        sa.Column('valid_until', sa.Date),
        sa.Column('status', sa.String(20), default='active'),
        sa.Column('meta', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        schema='cadastral'
    )
    
    # la_topology_tetra (TEN model)
    op.create_table(
        'la_topology_tetra',
        sa.Column('tet_id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('suid_1', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_spatial_unit.suid'), nullable=False, index=True),
        sa.Column('suid_2', UUID(as_uuid=True), sa.ForeignKey('cadastral.la_spatial_unit.suid'), nullable=False, index=True),
        sa.Column('shared_face', sa.Text),  # WKT TRIANGLE
        sa.Column('relationship', sa.String(20), nullable=False),
        sa.Column('metadata', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('suid_1', 'suid_2', name='ix_topo_pair'),
        schema='cadastral'
    )
    
    # ulpin_sequence
    op.create_table(
        'ulpin_sequence',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('component', sa.String(20), unique=True, nullable=False),
        sa.Column('current_value', sa.BigInteger, default=0),
        sa.Column('max_value', sa.BigInteger, nullable=False),
        schema='cadastral'
    )
    
    # Default ULPIN sequences
    op.execute("""
        INSERT INTO cadastral.ulpin_sequence (component, current_value, max_value) VALUES
        ('state', 1, 99),
        ('district', 1, 999),
        ('subdistrict', 1, 99),
        ('village', 1, 9999),
        ('parcel', 1, 999999),
        ('vertical', 0, 9999),
        ('unit', 0, 9999)
        ON CONFLICT (component) DO NOTHING;
    """)
    
    # 3D spatial index
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_spatial_unit_geom_3d 
        ON cadastral.la_spatial_unit USING GIST (ST_GeomFromText(geometry, 4326));
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_spatial_unit_z_bounds 
        ON cadastral.la_spatial_unit (z_min, z_max);
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_rrr_spatial_unit_valid 
        ON cadastral.la_rrr (spatial_unit_ref, valid_from, valid_until) 
        WHERE status = 'active';
    """)
    
    # ULPIN functions
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.generate_3d_ulpin(
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
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.parse_3d_ulpin(p_ulpin VARCHAR(25)) 
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
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.compute_floor_level(z_base DOUBLE PRECISION, z_top DOUBLE PRECISION, ground_z DOUBLE PRECISION DEFAULT 0)
        RETURNS INTEGER AS $$
        DECLARE
            floor_height CONSTANT DOUBLE PRECISION := 3.0;
            mid_z DOUBLE PRECISION := (z_base + z_top) / 2.0;
        BEGIN
            RETURN ROUND((mid_z - ground_z) / floor_height)::INTEGER;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Validation functions
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.validate_3d_adjacency()
        RETURNS TABLE(suid_1 UUID, suid_2 UUID, relation TEXT, details TEXT) AS $$
        BEGIN
            RETURN QUERY
            SELECT a.suid, b.suid, 
                   CASE 
                       WHEN ST_Overlaps(a.geometry, b.geometry) THEN 'overlap'
                       WHEN ST_Touches(a.geometry, b.geometry) THEN 'adjacent'
                       WHEN ST_Contains(a.geometry, b.geometry) THEN 'contain'
                       ELSE 'unknown'
                   END as relation,
                   '3D validation' as details
            FROM cadastral.la_spatial_unit a, cadastral.la_spatial_unit b
            WHERE a.suid < b.suid
              AND ST_3DIntersects(a.geometry, b.geometry)
              AND a.parent_suid = b.parent_suid;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.validate_vertical_continuity()
        RETURNS TABLE(upper_suid UUID, lower_suid UUID, gap_m DOUBLE PRECISION) AS $$
        BEGIN
            RETURN QUERY
            SELECT a.suid, b.suid, ST_ZMin(a.geometry) - ST_ZMax(b.geometry) as gap
            FROM cadastral.la_spatial_unit a, cadastral.la_spatial_unit b
            WHERE a.parent_suid = b.parent_suid
              AND a.floor_level = b.floor_level + 1
              AND ABS(ST_ZMin(a.geometry) - ST_ZMax(b.geometry)) > 0.01;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION cadastral.validate_boundary_alignment()
        RETURNS TABLE(unit_suid UUID, parcel_suid UUID, details TEXT) AS $$
        BEGIN
            RETURN QUERY
            SELECT u.suid, p.suid, 'Unit exceeds parcel footprint'
            FROM cadastral.la_spatial_unit u, cadastral.la_spatial_unit p
            WHERE u.parent_suid = p.suid
              AND NOT ST_Within(ST_Force2D(u.geometry), ST_Force2D(p.geometry));
        END;
        $$ LANGUAGE plpgsql;
    """)

def downgrade():
    op.execute('DROP FUNCTION IF EXISTS cadastral.validate_boundary_alignment();')
    op.execute('DROP FUNCTION IF EXISTS cadastral.validate_vertical_continuity();')
    op.execute('DROP FUNCTION IF EXISTS cadastral.validate_3d_adjacency();')
    op.execute('DROP FUNCTION IF EXISTS cadastral.compute_floor_level;')
    op.execute('DROP FUNCTION IF EXISTS cadastral.parse_3d_ulpin;')
    op.execute('DROP FUNCTION IF EXISTS cadastral.generate_3d_ulpin;')
    
    op.drop_table('ulpin_sequence', schema='cadastral')
    op.drop_table('la_topology_tetra', schema='cadastral')
    op.drop_table('la_rrr', schema='cadastral')
    op.drop_table('la_spatial_unit', schema='cadastral')
    op.drop_table('la_survey_source', schema='cadastral')
    op.drop_table('la_right', schema='cadastral')
    op.drop_table('la_party', schema='cadastral')
    
    op.execute('DROP EXTENSION IF EXISTS postgis_topology;')
    op.execute('DROP EXTENSION IF EXISTS postgis_raster;')
    op.execute('DROP EXTENSION IF EXISTS postgis;')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp";')