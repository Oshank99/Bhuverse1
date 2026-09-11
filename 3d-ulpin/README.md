# 3D ULPIN Cadastral System

> **National 3D Cadastral Framework** - Scalable, standards-compliant system for generating unique spatial identities (3D ULPINs) for surface parcels, multi-storey apartments, underground infrastructure, and air-rights.

## 🎯 Features

- **3D ULPIN Generation**: Deterministic 25-digit hierarchical identifiers (State-District-Village-Parcel-Vertical-Unit)
- **Multi-source Ingestion**: LiDAR (PDAL), Drone (SfM), Floor Plans (DXF/IFC), GNSS/CORS
- **AI/ML Pipeline**: Building extraction, floor segmentation, vertical delineation
- **3D Topology Validation**: TEN (Tetrahedral Network) model for gap/overlap detection
- **Standards Compliant**: ISO 19152 LADM, CityGML 3.0, OGC 3D Tiles, OGC API-Features
- **Web Viewer**: CesiumJS-based 3D cadastral explorer

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    3D ULPIN PLATFORM                        │
├─────────────────────────────────────────────────────────────┤
│  API Gateway (FastAPI)  │  Auth (Keycloak)  │  Monitoring  │
├─────────────────────────────────────────────────────────────┤
│  Ingestion Service      │  Processing Engine  │  3D ULPIN   │
│  - Drone/LiDAR/GNSS     │  - PDAL pipelines   │  - LADM     │
│  - Floor plans (DXF/IFC)│  - ML inference     │  - Topology │
│  - GIS layers (GPKG)    │  - TEN topology     │  - Version  │
├─────────────────────────────────────────────────────────────┤
│  PostGIS 3D (Primary)   │  MinIO/S3 (Objects) │  Redis      │
│  - LADM schema          │  - Point clouds     │  - Cache    │
│  - TEN topology         │  - 3D Tiles         │  - Sessions │
│  - 3D spatial index     │  - BIM models       │             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended

### Start System
```bash
# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

### Access Points
| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| 3D Viewer | http://localhost:80 |
| MinIO Console | http://localhost:9001 (minioadmin/minioadmin) |

## 📚 API Examples

### Generate 3D ULPIN
```bash
curl -X POST http://localhost:8000/ulpin/generate \
  -H "Content-Type: application/json" \
  -d '{"x": 77.209, "y": 28.614, "z": 213, "level": "unit"}'
```

### Search Spatial Units
```bash
curl "http://localhost:8000/collections/spatial_units/items?type=unit&floor_level=3&limit=10"
```

### Create Building from Footprint
```bash
curl -X POST http://localhost:8000/buildings/extrude \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "BLDG-001",
    "footprint_wkt": "POLYGON((77.2085 28.6135, 77.2095 28.6135, 77.2095 28.6145, 77.2085 28.6145, 77.2085 28.6135))",
    "num_floors": 10,
    "floor_height": 3.0,
    "ground_z": 210.0
  }'
```

### Validate Topology
```bash
curl -X POST http://localhost:8000/topology/validate \
  -H "Content-Type: application/json" \
  -d '{"units": [{"suid": "...", "vertices": [[...]]}], "tolerance": 0.01}'
```

## 🔧 Development

### Project Structure
```
3d-ulpin/
├── backend/
│   ├── api/           # FastAPI endpoints (OGC API-Features)
│   ├── core/          # Database, ULPIN, Topology
│   ├── ingestion/     # PDAL, DXF/IFC, Vertical Delineation
│   └── ml/            # Model serving
├── frontend/          # CesiumJS viewer
├── alembic/           # Database migrations
├── infra/             # Docker, Nginx, DB init
└── docker-compose.yml
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start database only
docker-compose up -d postgres minio redis

# Run migrations
alembic upgrade head

# Start API with hot reload
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker
celery -A backend.ingestion.celery_app worker --loglevel=info
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📐 3D ULPIN Format

```
┌─────────────────────────────────────────────────────────────┐
│  [State(2)][District(3)][SubDist(2)][Village(4)][Parcel(6)] │
│  [Vertical(4)][Unit(4)] = 25 digits                         │
└─────────────────────────────────────────────────────────────┘

Example: 0100101000100000100010001
         │   │   │   │    │    │  │
         │   │   │   │    │    │  └── Unit: 0001 (Apt 1)
         │   │   │   │    │    └───── Vertical: 0001 (Floor 1)
         │   │   │   │    └────────── Parcel: 000001
         │   │   │   └─────────────── Village: 0001
         │   │   └──────────────────── SubDistrict: 01
         │   └──────────────────────── District: 001
         └──────────────────────────── State: 01
```

**Vertical Encoding**: Floor level +100 offset (Ground=100, Floor 1=101, Basement=-1=099)

## 🏛️ Standards Compliance

| Standard | Implementation |
|----------|----------------|
| **ISO 19152 LADM** | Core data model (Parties, Rights, RRRs, Spatial Units) |
| **CityGML 3.0** | Building semantics, 3D geometry exchange |
| **OGC 3D Tiles 1.1** | Streaming visualization (B3DM/glTF) |
| **OGC API-Features** | RESTful feature access |
| **ISO 19107** | GM_Solid, GM_MultiSolid geometry |
| **ASPRS LAS 1.4** | LiDAR point cloud interchange |

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v --cov=backend

# Specific test modules
pytest tests/test_ulpin.py -v
pytest tests/test_topology.py -v
pytest tests/test_ingestion.py -v
```

## 📊 Performance Targets

| Metric | MVP Target | Production Target |
|--------|------------|-------------------|
| Ingestion throughput | 1 km²/day | 50 km²/day |
| ULPIN generation | <100ms | <50ms |
| Topology validation (1000 units) | <5 min | <1 min |
| ML building extraction F1 | >0.80 | >0.90 |
| API response (tile query) | <500ms | <200ms |

## 🔐 Security

- JWT-based authentication (Keycloak integration ready)
- Role-based access control (Admin, Surveyor, Viewer)
- Audit logging for all ULPIN assignments
- API rate limiting
- Data encryption at rest (PostgreSQL TDE) and in transit (TLS)

## 🌍 Deployment

### Kubernetes (Production)
```bash
# Apply manifests
kubectl apply -f infra/k8s/

# Scale workers
kubectl scale deployment ulpin-worker --replicas=5
```

### Environment Variables
See `.env` for all configuration options.

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Documentation**: `/docs` endpoint
- **Issues**: GitHub Issues
- **Standards**: OGC/ISO working groups

---

*Built for the Digital India Land Records Modernization Programme (DILRMP)*
*Compatible with ULPIN 14-digit → 3D ULPIN 25-digit extension*