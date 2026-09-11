# 3D ULPIN System - Docker Configuration
FROM python:3.11-slim

# Install system dependencies for GDAL/PDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    pdal \
    libpdal-dev \
    libpdal-plugin-* \
    postgresql-client \
    libpq-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL config
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]