@echo off
REM 3D ULPIN System - Quick Start Script (Windows)

echo 🚀 Starting 3D ULPIN Cadastral System...

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker first.
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose not found. Please install Docker Compose first.
    exit /b 1
)

REM Create necessary directories
if not exist backend\ml\models mkdir backend\ml\models

REM Start services
echo 📦 Starting Docker services...
docker-compose up -d

REM Wait for database
echo ⏳ Waiting for database...
timeout /t 10 >nul

REM Check API health
echo 🔍 Checking API health...
for /l %%i in (1,1,30) do (
    curl -s http://localhost:8000/health >nul
    if not errorlevel 1 (
        echo ✅ API is healthy!
        goto :health_ok
    )
    timeout /t 2 >nul
)
:health_ok

REM Load sample data
echo 📊 Loading sample data...
curl -X POST http://localhost:8000/buildings/extrude ^
  -H "Content-Type: application/json" ^
  -d "{\"building_id\": \"DEMO-001\", \"footprint_wkt\": \"POLYGON((77.2085 28.6135, 77.2095 28.6135, 77.2095 28.6145, 77.2085 28.6145, 77.2085 28.6135))\", \"num_floors\": 5, \"floor_height\": 3.0, \"ground_z\": 210.0, \"unit_layout\": \"grid\"}"

echo.
echo ✨ 3D ULPIN System is running!
echo.
echo 🌐 Access points:
echo    API:        http://localhost:8000
echo    Docs:       http://localhost:8000/docs
echo    Viewer:     http://localhost:80
echo    MinIO:      http://localhost:9001 (minioadmin/minioadmin)
echo.
echo 📖 Try these API calls:
echo    curl http://localhost:8000/collections
echo    curl http://localhost:8000/collections/spatial_units/items
echo    curl -X POST http://localhost:8000/ulpin/generate -H "Content-Type: application/json" -d "{\"x\": 77.209, \"y\": 28.614, \"z\": 213}"
echo.
echo 🛑 To stop: docker-compose down
echo 🗑️  To reset: docker-compose down -v