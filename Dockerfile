# Multi-Stage / Slim Python Dockerfile for Render Deployment
# Project ORCA: ISRO SIH 176 Marine Multi-Agent System

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system libraries required for HDF5, NetCDF, and Geospatial calculations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create cache and data directories with appropriate permissions
RUN mkdir -p data/mosdac_cache data/copernicus_cache logs

# Expose default port
EXPOSE 8000

# Render dynamically injects $PORT; bind to 0.0.0.0 using ${PORT:-8000}
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]

