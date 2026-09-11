-- 3D ULPIN Database Initialization
-- Run on PostgreSQL with PostGIS

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS cadastral;
CREATE SCHEMA IF NOT EXISTS topology;

-- Set search path
SET search_path TO cadastral, topology, public;

-- 3D ULPIN generator function
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

-- Parse ULPIN
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

-- Floor level from Z
CREATE OR REPLACE FUNCTION cadastral.compute_floor_level(z_base DOUBLE PRECISION, z_top DOUBLE PRECISION, ground_z DOUBLE PRECISION DEFAULT 0)
RETURNS INTEGER AS $$
DECLARE
    floor_height CONSTANT DOUBLE PRECISION := 3.0;
    mid_z DOUBLE PRECISION := (z_base + z_top) / 2.0;
BEGIN
    RETURN ROUND((mid_z - ground_z) / floor_height)::INTEGER;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 3D topology validation functions
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

-- Vertical continuity check
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

-- Boundary alignment check
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_spatial_unit_geom_3d 
ON cadastral.la_spatial_unit USING GIST (ST_GeomFromText(geometry, 4326));

CREATE INDEX IF NOT EXISTS idx_spatial_unit_uln ON cadastral.la_spatial_unit (uln);
CREATE INDEX IF NOT EXISTS idx_spatial_unit_type ON cadastral.la_spatial_unit (type);
CREATE INDEX IF NOT EXISTS idx_spatial_unit_floor ON cadastral.la_spatial_unit (floor_level);
CREATE INDEX IF NOT EXISTS idx_spatial_unit_parent ON cadastral.la_spatial_unit (parent_suid);
CREATE INDEX IF NOT EXISTS idx_spatial_unit_z ON cadastral.la_spatial_unit (z_min, z_max);

CREATE INDEX IF NOT EXISTS idx_rrr_spatial_active 
ON cadastral.la_rrr (spatial_unit_ref, valid_from, valid_until) 
WHERE status = 'active';

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA cadastral TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA topology TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cadastral TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA topology TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cadastral TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA topology TO postgres;

-- Default admin hierarchy (India example)
INSERT INTO cadastral.ulpin_sequence (component, current_value, max_value) VALUES
('state', 1, 99),
('district', 1, 999),
('subdistrict', 1, 99),
('village', 1, 9999),
('parcel', 1, 999999),
('vertical', 0, 9999),
('unit', 0, 9999)
ON CONFLICT (component) DO NOTHING;