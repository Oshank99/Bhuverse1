"""
Celery app for background ingestion tasks
"""
from celery import Celery
import os

celery_app = Celery('ulpin3d')

celery_app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'backend.ingestion.tasks.process_lidar': {'queue': 'ml'},
        'backend.ingestion.tasks.process_floorplan': {'queue': 'default'},
        'backend.ingestion.tasks.process_drone': {'queue': 'ml'},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['backend.ingestion.tasks'])