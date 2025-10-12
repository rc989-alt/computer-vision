#!/usr/bin/env python3
"""
Production Configuration & Deployment Scripts

Environment-specific configurations for:
- Laptop development (4-item GPU batches, local Redis)
- Cloud production (512-item GPU batches, managed services)
- Staging environment (mixed configuration)

Includes deployment automation and environment switching.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml
import subprocess
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    description: str
    
    # Processing configuration
    batch_size: int
    max_workers: int
    gpu_memory_gb: float
    
    # Queue configuration
    queue_backend: str  # redis, kafka, sqs
    queue_config: Dict[str, Any]
    
    # Cache configuration
    cache_backend: str  # redis, memcached
    cache_config: Dict[str, Any]
    
    # Storage configuration
    storage_backend: str  # local, s3, gcs
    storage_config: Dict[str, Any]
    
    # Monitoring configuration
    monitoring_enabled: bool
    monitoring_config: Dict[str, Any]
    
    # Resource limits
    cpu_limit: str
    memory_limit: str
    disk_limit: str

class ConfigurationManager:
    """Manages environment configurations"""
    
    def __init__(self, config_dir: str = "config/environments"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.environments = self._load_environments()
        
    def _load_environments(self) -> Dict[str, EnvironmentConfig]:
        """Load all environment configurations"""
        environments = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    environments[config_data['name']] = EnvironmentConfig(**config_data)
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")
                
        return environments
    
    def get_environment(self, name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name"""
        return self.environments.get(name)
    
    def create_environment(self, config: EnvironmentConfig):
        """Create new environment configuration"""
        config_path = self.config_dir / f"{config.name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
        
        self.environments[config.name] = config
        logger.info(f"Created environment config: {config.name}")
    
    def list_environments(self) -> Dict[str, str]:
        """List available environments with descriptions"""
        return {name: env.description for name, env in self.environments.items()}

def create_laptop_config() -> EnvironmentConfig:
    """Create laptop development configuration"""
    return EnvironmentConfig(
        name="laptop",
        description="Laptop development environment with GPU optimization",
        
        # Processing - laptop GPU friendly
        batch_size=4,
        max_workers=3,
        gpu_memory_gb=8.0,
        
        # Queue - local Redis
        queue_backend="redis",
        queue_config={
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "max_connections": 10,
            "socket_timeout": 30
        },
        
        # Cache - local Redis
        cache_backend="redis", 
        cache_config={
            "host": "localhost",
            "port": 6379,
            "db": 1,
            "max_memory": "1gb",
            "eviction_policy": "allkeys-lru"
        },
        
        # Storage - local filesystem
        storage_backend="local",
        storage_config={
            "base_path": "./data",
            "temp_path": "/tmp/pipeline",
            "backup_enabled": False
        },
        
        # Monitoring - minimal
        monitoring_enabled=True,
        monitoring_config={
            "metrics_retention": "24h",
            "alert_webhook": None,
            "dashboard_enabled": True,
            "log_level": "INFO"
        },
        
        # Resource limits - laptop friendly
        cpu_limit="4",
        memory_limit="8Gi",
        disk_limit="100Gi"
    )

def create_cloud_production_config() -> EnvironmentConfig:
    """Create cloud production configuration"""
    return EnvironmentConfig(
        name="production",
        description="Cloud production environment with high throughput",
        
        # Processing - cloud GPU optimized
        batch_size=512,
        max_workers=8,
        gpu_memory_gb=32.0,
        
        # Queue - managed Kafka/SQS
        queue_backend="kafka",
        queue_config={
            "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
            "sasl_username": "${KAFKA_USERNAME}",
            "sasl_password": "${KAFKA_PASSWORD}",
            "partitions": 16,
            "replication_factor": 3
        },
        
        # Cache - managed Redis cluster
        cache_backend="redis",
        cache_config={
            "cluster_endpoint": "redis-cluster.cache.amazonaws.com:6379",
            "ssl": True,
            "max_connections": 100,
            "max_memory": "16gb",
            "eviction_policy": "allkeys-lru"
        },
        
        # Storage - cloud object storage
        storage_backend="s3",
        storage_config={
            "bucket": "pipeline-data-prod",
            "region": "us-west-2",
            "encryption": "AES256",
            "versioning": True,
            "lifecycle_rules": {
                "logs": {"expiration": 90},
                "temp": {"expiration": 7}
            }
        },
        
        # Monitoring - full observability
        monitoring_enabled=True,
        monitoring_config={
            "metrics_retention": "30d",
            "alert_webhook": "https://hooks.slack.com/services/...",
            "dashboard_enabled": True,
            "log_level": "INFO",
            "distributed_tracing": True,
            "metrics_export": {
                "prometheus": True,
                "cloudwatch": True
            }
        },
        
        # Resource limits - production scale
        cpu_limit="16",
        memory_limit="64Gi", 
        disk_limit="1Ti"
    )

def create_staging_config() -> EnvironmentConfig:
    """Create staging environment configuration"""
    return EnvironmentConfig(
        name="staging",
        description="Staging environment for testing production configurations",
        
        # Processing - medium scale
        batch_size=32,
        max_workers=4,
        gpu_memory_gb=16.0,
        
        # Queue - local Redis with clustering
        queue_backend="redis",
        queue_config={
            "cluster_enabled": True,
            "nodes": [
                {"host": "redis-1", "port": 6379},
                {"host": "redis-2", "port": 6379},
                {"host": "redis-3", "port": 6379}
            ],
            "max_connections": 20
        },
        
        # Cache - Redis cluster
        cache_backend="redis",
        cache_config={
            "cluster_enabled": True,
            "nodes": [
                {"host": "redis-1", "port": 6380},
                {"host": "redis-2", "port": 6380},
                {"host": "redis-3", "port": 6380}
            ],
            "max_memory": "4gb"
        },
        
        # Storage - MinIO (S3-compatible)
        storage_backend="s3",
        storage_config={
            "endpoint": "http://minio:9000",
            "bucket": "pipeline-data-staging",
            "access_key": "${MINIO_ACCESS_KEY}",
            "secret_key": "${MINIO_SECRET_KEY}",
            "secure": False
        },
        
        # Monitoring - production-like
        monitoring_enabled=True,
        monitoring_config={
            "metrics_retention": "7d",
            "alert_webhook": "https://hooks.slack.com/services/.../staging",
            "dashboard_enabled": True,
            "log_level": "DEBUG",
            "distributed_tracing": True
        },
        
        # Resource limits - medium scale
        cpu_limit="8",
        memory_limit="32Gi",
        disk_limit="500Gi"
    )

class DockerComposeGenerator:
    """Generates Docker Compose configurations for different environments"""
    
    @staticmethod
    def generate_laptop_compose(config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate Docker Compose for laptop environment"""
        return {
            'version': '3.8',
            'services': {
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'command': 'redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru',
                    'volumes': ['redis_data:/data']
                },
                'pipeline': {
                    'build': '.',
                    'depends_on': ['redis'],
                    'environment': {
                        'ENVIRONMENT': 'laptop',
                        'REDIS_URL': 'redis://redis:6379',
                        'BATCH_SIZE': str(config.batch_size),
                        'GPU_MEMORY_GB': str(config.gpu_memory_gb)
                    },
                    'volumes': [
                        './data:/app/data',
                        './config:/app/config',
                        './logs:/app/logs'
                    ],
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': config.cpu_limit,
                                'memory': config.memory_limit
                            },
                            'reservations': {
                                'devices': [{
                                    'driver': 'nvidia',
                                    'count': 1,
                                    'capabilities': ['gpu']
                                }]
                            }
                        }
                    }
                }
            },
            'volumes': {
                'redis_data': {}
            }
        }
    
    @staticmethod 
    def generate_staging_compose(config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate Docker Compose for staging environment"""
        return {
            'version': '3.8',
            'services': {
                'redis-1': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379', '6380:6380'],
                    'command': 'redis-server --port 6379 --cluster-enabled yes --cluster-config-file nodes.conf',
                    'volumes': ['redis1_data:/data']
                },
                'redis-2': {
                    'image': 'redis:7-alpine', 
                    'ports': ['6381:6379', '6382:6380'],
                    'command': 'redis-server --port 6379 --cluster-enabled yes --cluster-config-file nodes.conf',
                    'volumes': ['redis2_data:/data']
                },
                'redis-3': {
                    'image': 'redis:7-alpine',
                    'ports': ['6383:6379', '6384:6380'],
                    'command': 'redis-server --port 6379 --cluster-enabled yes --cluster-config-file nodes.conf',
                    'volumes': ['redis3_data:/data']
                },
                'minio': {
                    'image': 'minio/minio:latest',
                    'ports': ['9000:9000', '9001:9001'],
                    'environment': {
                        'MINIO_ACCESS_KEY': 'pipeline_access',
                        'MINIO_SECRET_KEY': 'pipeline_secret_key_123'
                    },
                    'command': 'server /data --console-address ":9001"',
                    'volumes': ['minio_data:/data']
                },
                'pipeline': {
                    'build': '.',
                    'depends_on': ['redis-1', 'redis-2', 'redis-3', 'minio'],
                    'environment': {
                        'ENVIRONMENT': 'staging',
                        'REDIS_CLUSTER': 'redis-1:6379,redis-2:6381,redis-3:6383',
                        'MINIO_ENDPOINT': 'http://minio:9000',
                        'MINIO_ACCESS_KEY': 'pipeline_access',
                        'MINIO_SECRET_KEY': 'pipeline_secret_key_123',
                        'BATCH_SIZE': str(config.batch_size),
                        'GPU_MEMORY_GB': str(config.gpu_memory_gb)
                    },
                    'volumes': [
                        './config:/app/config',
                        './logs:/app/logs'
                    ],
                    'deploy': {
                        'replicas': 2,
                        'resources': {
                            'limits': {
                                'cpus': config.cpu_limit,
                                'memory': config.memory_limit
                            }
                        }
                    }
                }
            },
            'volumes': {
                'redis1_data': {},
                'redis2_data': {},
                'redis3_data': {},
                'minio_data': {}
            }
        }

class KubernetesGenerator:
    """Generates Kubernetes manifests for production deployment"""
    
    @staticmethod
    def generate_production_manifests(config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate Kubernetes manifests for production"""
        
        # ConfigMap for environment configuration
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'pipeline-config',
                'namespace': 'production'
            },
            'data': {
                'environment': 'production',
                'batch_size': str(config.batch_size),
                'max_workers': str(config.max_workers),
                'gpu_memory_gb': str(config.gpu_memory_gb)
            }
        }
        
        # Deployment for pipeline workers
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'pipeline-workers',
                'namespace': 'production'
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'pipeline-workers'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'pipeline-workers'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'pipeline-worker',
                            'image': 'your-registry/pipeline:latest',
                            'env': [
                                {
                                    'name': 'ENVIRONMENT',
                                    'valueFrom': {
                                        'configMapKeyRef': {
                                            'name': 'pipeline-config',
                                            'key': 'environment'
                                        }
                                    }
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': '2',
                                    'memory': '8Gi',
                                    'nvidia.com/gpu': '1'
                                },
                                'limits': {
                                    'cpu': config.cpu_limit,
                                    'memory': config.memory_limit,
                                    'nvidia.com/gpu': '1'
                                }
                            },
                            'ports': [{
                                'containerPort': 8080,
                                'name': 'metrics'
                            }]
                        }],
                        'nodeSelector': {
                            'node-type': 'gpu-worker'
                        },
                        'tolerations': [{
                            'key': 'nvidia.com/gpu',
                            'operator': 'Exists',
                            'effect': 'NoSchedule'
                        }]
                    }
                }
            }
        }
        
        # HorizontalPodAutoscaler
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'pipeline-workers-hpa',
                'namespace': 'production'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'pipeline-workers'
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Pods',
                        'pods': {
                            'metric': {
                                'name': 'queue_depth'
                            },
                            'target': {
                                'type': 'AverageValue',
                                'averageValue': '50'
                            }
                        }
                    }
                ]
            }
        }
        
        return {
            'configmap': configmap,
            'deployment': deployment,
            'hpa': hpa
        }

class DeploymentManager:
    """Manages deployment across environments"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        
    def deploy_laptop(self, config_name: str = "laptop"):
        """Deploy to laptop environment"""
        config = self.config_manager.get_environment(config_name)
        if not config:
            raise ValueError(f"Environment not found: {config_name}")
        
        logger.info(f"Deploying to laptop environment...")
        
        # Generate Docker Compose
        compose_config = DockerComposeGenerator.generate_laptop_compose(config)
        
        # Write compose file
        with open('docker-compose.laptop.yml', 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        # Start services
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.laptop.yml', 'up', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Laptop deployment successful")
            print("🚀 Laptop environment ready!")
            print("   Redis: localhost:6379")
            print("   Pipeline: running in container")
            print("   Logs: ./logs/")
        else:
            logger.error(f"Deployment failed: {result.stderr}")
            raise RuntimeError(f"Docker Compose failed: {result.stderr}")
    
    def deploy_staging(self, config_name: str = "staging"):
        """Deploy to staging environment"""
        config = self.config_manager.get_environment(config_name)
        if not config:
            raise ValueError(f"Environment not found: {config_name}")
        
        logger.info(f"Deploying to staging environment...")
        
        # Generate Docker Compose
        compose_config = DockerComposeGenerator.generate_staging_compose(config)
        
        # Write compose file
        with open('docker-compose.staging.yml', 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        # Start services
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.staging.yml', 'up', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Staging deployment successful")
            print("🚀 Staging environment ready!")
            print("   Redis Cluster: localhost:6379,6381,6383")
            print("   MinIO: localhost:9000")
            print("   Pipeline: 2 replicas running")
        else:
            logger.error(f"Deployment failed: {result.stderr}")
            raise RuntimeError(f"Docker Compose failed: {result.stderr}")
    
    def generate_production_manifests(self, config_name: str = "production", output_dir: str = "k8s"):
        """Generate Kubernetes manifests for production"""
        config = self.config_manager.get_environment(config_name)
        if not config:
            raise ValueError(f"Environment not found: {config_name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate manifests
        manifests = KubernetesGenerator.generate_production_manifests(config)
        
        # Write manifest files
        for name, manifest in manifests.items():
            manifest_path = output_path / f"{name}.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Generated Kubernetes manifests in {output_dir}/")
        print(f"📁 Kubernetes manifests generated:")
        for name in manifests.keys():
            print(f"   {output_dir}/{name}.yaml")
        
        print(f"\n🚀 To deploy to production:")
        print(f"   kubectl apply -f {output_dir}/")

def main():
    """Setup production configurations and deployment"""
    print("⚙️  PRODUCTION CONFIGURATION & DEPLOYMENT")
    print("=" * 50)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Create environment configurations
    laptop_config = create_laptop_config()
    staging_config = create_staging_config()
    production_config = create_cloud_production_config()
    
    # Save configurations
    config_manager.create_environment(laptop_config)
    config_manager.create_environment(staging_config)
    config_manager.create_environment(production_config)
    
    print("📝 Environment configurations created:")
    for name, description in config_manager.list_environments().items():
        print(f"   • {name}: {description}")
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(config_manager)
    
    # Generate deployment files
    print(f"\n🐳 Generating Docker Compose files...")
    
    laptop_compose = DockerComposeGenerator.generate_laptop_compose(laptop_config)
    with open('docker-compose.laptop.yml', 'w') as f:
        yaml.dump(laptop_compose, f, default_flow_style=False, indent=2)
    
    staging_compose = DockerComposeGenerator.generate_staging_compose(staging_config)
    with open('docker-compose.staging.yml', 'w') as f:
        yaml.dump(staging_compose, f, default_flow_style=False, indent=2)
    
    print("✅ Docker Compose files generated")
    
    # Generate Kubernetes manifests
    print(f"\n☸️  Generating Kubernetes manifests...")
    deployment_manager.generate_production_manifests()
    
    print(f"\n🚀 DEPLOYMENT READY!")
    print("=" * 30)
    print("🖥️  Laptop: docker-compose -f docker-compose.laptop.yml up -d")
    print("🧪 Staging: docker-compose -f docker-compose.staging.yml up -d")
    print("☁️  Production: kubectl apply -f k8s/")
    
    print(f"\n📊 Configuration Summary:")
    print(f"   Laptop: {laptop_config.batch_size} batch, {laptop_config.gpu_memory_gb}GB GPU")
    print(f"   Staging: {staging_config.batch_size} batch, {staging_config.gpu_memory_gb}GB GPU")
    print(f"   Production: {production_config.batch_size} batch, {production_config.gpu_memory_gb}GB GPU")

if __name__ == "__main__":
    main()