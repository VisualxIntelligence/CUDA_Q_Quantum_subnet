"""
CUDA-Q Configuration Management and Integration Utilities
Provides configuration, installation, and integration tools for CUDA-Q solver
"""

import os
import json
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

import bittensor as bt


@dataclass
class CudaQInstallationConfig:
    """Configuration for CUDA-Q installation"""
    install_method: str = "pip"  # "pip", "conda", "docker", "source"
    version: str = "latest"
    cuda_version: str = "12.0"
    python_version: str = "3.11"
    install_path: Optional[str] = None
    verify_installation: bool = True


@dataclass
class CudaQPerformanceConfig:
    """Performance-related configuration"""
    enable_profiling: bool = False
    memory_optimization: bool = True
    cache_kernels: bool = True
    parallel_compilation: bool = True
    optimization_level: int = 2


@dataclass
class CudaQLoggingConfig:
    """Logging configuration for CUDA-Q operations"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_cuda_q_operations: bool = True
    log_performance_metrics: bool = True
    log_backend_selection: bool = True


@dataclass
class CudaQIntegrationConfig:
    """Complete CUDA-Q integration configuration"""
    # Core settings
    enabled: bool = True
    fallback_to_qiskit: bool = True
    auto_detect_hardware: bool = True
    
    # Backend preferences
    nvidia_enabled: bool = True
    mgpu_enabled: bool = True
    mqpu_enabled: bool = True
    remote_mqpu_enabled: bool = False
    
    # Performance thresholds
    small_circuit_qubits: int = 20
    medium_circuit_qubits: int = 30
    large_circuit_qubits: int = 40
    memory_pooling_threshold_gb: float = 8.0
    
    # Execution settings
    max_parallel_circuits: int = 1000
    batch_optimization: bool = True
    max_memory_gb: float = 320.0
    default_shots: int = 1000
    
    # Advanced settings
    performance: CudaQPerformanceConfig = None
    logging: CudaQLoggingConfig = None
    installation: CudaQInstallationConfig = None
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = CudaQPerformanceConfig()
        if self.logging is None:
            self.logging = CudaQLoggingConfig()
        if self.installation is None:
            self.installation = CudaQInstallationConfig()


class CudaQInstaller:
    """Handles CUDA-Q installation and setup"""
    
    @staticmethod
    def check_cuda_q_availability() -> bool:
        """Check if CUDA-Q is available in the current environment"""
        try:
            import cudaq
            return True
        except ImportError:
            return False
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """Check GPU availability and CUDA installation"""
        gpu_info = {
            "cuda_available": False,
            "num_gpus": 0,
            "gpu_names": [],
            "cuda_version": None,
            "driver_version": None
        }
        
        try:
            # Check NVIDIA-SMI
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", 
                                   "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info["num_gpus"] = len(lines)
                gpu_info["gpu_names"] = [line.split(',')[0].strip() for line in lines]
                gpu_info["cuda_available"] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            # Check CUDA version
            result = subprocess.run(["nvcc", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "release" in line.lower():
                        gpu_info["cuda_version"] = line.split()[-1]
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return gpu_info
    
    @staticmethod
    def install_cuda_q(config: CudaQInstallationConfig) -> bool:
        """Install CUDA-Q based on configuration"""
        bt.logging.info(f"Installing CUDA-Q using method: {config.install_method}")
        
        if config.install_method == "pip":
            return CudaQInstaller._install_via_pip(config)
        elif config.install_method == "conda":
            return CudaQInstaller._install_via_conda(config)
        elif config.install_method == "docker":
            return CudaQInstaller._install_via_docker(config)
        else:
            bt.logging.error(f"Unsupported installation method: {config.install_method}")
            return False
    
    @staticmethod
    def _install_via_pip(config: CudaQInstallationConfig) -> bool:
        """Install CUDA-Q via pip"""
        try:
            # Install CUDA-Q
            if config.version == "latest":
                cmd = [sys.executable, "-m", "pip", "install", "cuda-quantum"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", f"cuda-quantum=={config.version}"]
            
            bt.logging.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                bt.logging.info("CUDA-Q installed successfully via pip")
                return config.verify_installation and CudaQInstaller._verify_installation()
            else:
                bt.logging.error(f"Failed to install CUDA-Q: {result.stderr}")
                return False
                
        except Exception as e:
            bt.logging.error(f"Exception during pip installation: {e}")
            return False
    
    @staticmethod
    def _install_via_conda(config: CudaQInstallationConfig) -> bool:
        """Install CUDA-Q via conda"""
        try:
            # Check if conda is available
            subprocess.run(["conda", "--version"], capture_output=True, timeout=5)
            
            # Install CUDA-Q
            cmd = ["conda", "install", "-c", "nvidia", "cuda-quantum", "-y"]
            if config.version != "latest":
                cmd[-2] = f"cuda-quantum={config.version}"
            
            bt.logging.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                bt.logging.info("CUDA-Q installed successfully via conda")
                return config.verify_installation and CudaQInstaller._verify_installation()
            else:
                bt.logging.error(f"Failed to install CUDA-Q via conda: {result.stderr}")
                return False
                
        except Exception as e:
            bt.logging.error(f"Exception during conda installation: {e}")
            return False
    
    @staticmethod
    def _install_via_docker(config: CudaQInstallationConfig) -> bool:
        """Setup CUDA-Q via Docker"""
        bt.logging.info("Docker installation method not implemented yet")
        return False
    
    @staticmethod
    def _verify_installation() -> bool:
        """Verify CUDA-Q installation"""
        try:
            import cudaq
            
            # Test basic functionality
            kernel = cudaq.make_kernel()
            qubit = kernel.qalloc()
            kernel.h(qubit)
            kernel.mz(qubit)
            
            # Try to get available targets
            targets = cudaq.get_targets()
            bt.logging.info(f"CUDA-Q verification successful. Available targets: {targets}")
            return True
            
        except Exception as e:
            bt.logging.error(f"CUDA-Q verification failed: {e}")
            return False


class CudaQConfigManager:
    """Manages CUDA-Q configuration files and settings"""
    
    DEFAULT_CONFIG_PATHS = [
        "cuda_q_config.yaml",
        "cuda_q_config.json",
        "~/.cuda_q/config.yaml",
        "/etc/cuda_q/config.yaml"
    ]
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> CudaQIntegrationConfig:
        """Load configuration from file or use defaults"""
        if config_path:
            return CudaQConfigManager._load_from_file(config_path)
        
        # Try default paths
        for path in CudaQConfigManager.DEFAULT_CONFIG_PATHS:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                return CudaQConfigManager._load_from_file(str(expanded_path))
        
        # Return default configuration
        bt.logging.info("No configuration file found, using defaults")
        return CudaQIntegrationConfig()
    
    @staticmethod
    def _load_from_file(config_path: str) -> CudaQIntegrationConfig:
        """Load configuration from specific file"""
        path = Path(config_path)
        
        if not path.exists():
            bt.logging.warning(f"Configuration file not found: {config_path}")
            return CudaQIntegrationConfig()
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    bt.logging.error(f"Unsupported config file format: {path.suffix}")
                    return CudaQIntegrationConfig()
            
            return CudaQConfigManager._dict_to_config(data)
            
        except Exception as e:
            bt.logging.error(f"Failed to load configuration from {config_path}: {e}")
            return CudaQIntegrationConfig()
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> CudaQIntegrationConfig:
        """Convert dictionary to configuration object"""
        try:
            # Handle nested configurations
            if 'performance' in data and isinstance(data['performance'], dict):
                data['performance'] = CudaQPerformanceConfig(**data['performance'])
            
            if 'logging' in data and isinstance(data['logging'], dict):
                data['logging'] = CudaQLoggingConfig(**data['logging'])
            
            if 'installation' in data and isinstance(data['installation'], dict):
                data['installation'] = CudaQInstallationConfig(**data['installation'])
            
            return CudaQIntegrationConfig(**data)
            
        except Exception as e:
            bt.logging.error(f"Failed to parse configuration: {e}")
            return CudaQIntegrationConfig()
    
    @staticmethod
    def save_config(config: CudaQIntegrationConfig, config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = asdict(config)
            
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                elif path.suffix.lower() == '.json':
                    json.dump(data, f, indent=2)
                else:
                    bt.logging.error(f"Unsupported config file format: {path.suffix}")
                    return
            
            bt.logging.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            bt.logging.error(f"Failed to save configuration to {config_path}: {e}")
    
    @staticmethod
    def create_default_config(config_path: str):
        """Create a default configuration file"""
        config = CudaQIntegrationConfig()
        CudaQConfigManager.save_config(config, config_path)


class CudaQIntegrationHelper:
    """Helper class for integrating CUDA-Q with existing quantum subnet code"""
    
    @staticmethod
    def create_solver_replacement_script(output_path: str):
        """Create a script to replace the default solver with CUDA-Q solver"""
        script_content = '''#!/usr/bin/env python3
"""
Script to replace default peaked solver with CUDA-Q enhanced solver
Run this script in your quantum subnet directory to enable CUDA-Q acceleration
"""

import os
import shutil
from pathlib import Path

def backup_original_solver():
    """Backup the original solver"""
    solver_path = Path("qbittensor/miner/solvers/default_peaked_solver.py")
    if solver_path.exists():
        backup_path = solver_path.with_suffix(".py.backup")
        shutil.copy2(solver_path, backup_path)
        print(f"Original solver backed up to {backup_path}")
        return True
    else:
        print("Original solver not found")
        return False

def install_cuda_q_solver():
    """Install the CUDA-Q enhanced solver"""
    # Copy the CUDA-Q solver to the appropriate location
    cuda_q_solver_path = Path("cuda_q_peaked_solver.py")
    target_path = Path("qbittensor/miner/solvers/cuda_q_peaked_solver.py")
    
    if cuda_q_solver_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cuda_q_solver_path, target_path)
        print(f"CUDA-Q solver installed to {target_path}")
        return True
    else:
        print("CUDA-Q solver file not found")
        return False

def update_solver_imports():
    """Update imports to use CUDA-Q solver"""
    # Update miner.py to import CUDA-Q solver
    miner_path = Path("qbittensor/miner/miner.py")
    if miner_path.exists():
        with open(miner_path, 'r') as f:
            content = f.read()
        
        # Replace import
        content = content.replace(
            "from .solvers.default_peaked_solver import DefaultPeakedSolver",
            "from .solvers.cuda_q_peaked_solver import CudaQPeakedSolver as DefaultPeakedSolver"
        )
        
        with open(miner_path, 'w') as f:
            f.write(content)
        
        print("Updated miner.py imports")
        return True
    else:
        print("miner.py not found")
        return False

def main():
    print("Installing CUDA-Q enhanced solver...")
    
    if backup_original_solver():
        if install_cuda_q_solver():
            if update_solver_imports():
                print("\\nCUDA-Q solver installation completed successfully!")
                print("\\nNext steps:")
                print("1. Install CUDA-Q: pip install cuda-quantum")
                print("2. Configure your settings in cuda_q_config.yaml")
                print("3. Restart your miner")
            else:
                print("Failed to update imports")
        else:
            print("Failed to install CUDA-Q solver")
    else:
        print("Failed to backup original solver")

if __name__ == "__main__":
    main()
'''
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_path, 0o755)
        bt.logging.info(f"Solver replacement script created: {output_path}")
    
    @staticmethod
    def create_installation_script(output_path: str):
        """Create a comprehensive installation script"""
        script_content = '''#!/bin/bash
# CUDA-Q Installation Script for Quantum Subnet

set -e

echo "CUDA-Q Installation Script for Quantum Subnet"
echo "=============================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for safety reasons."
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if ! command_exists pip3; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Check NVIDIA drivers
if ! command_exists nvidia-smi; then
    echo "Warning: nvidia-smi not found. CUDA-Q will fall back to CPU simulation."
    echo "For GPU acceleration, please install NVIDIA drivers and CUDA toolkit."
else
    echo "NVIDIA drivers detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Install CUDA-Q
echo "Installing CUDA-Q..."
pip3 install cuda-quantum

# Verify installation
echo "Verifying CUDA-Q installation..."
python3 -c "import cudaq; print(f'CUDA-Q version: {cudaq.__version__}'); print(f'Available targets: {cudaq.get_targets()}')"

# Create default configuration
echo "Creating default configuration..."
python3 -c "
from cuda_q_config import CudaQConfigManager
CudaQConfigManager.create_default_config('cuda_q_config.yaml')
print('Default configuration created: cuda_q_config.yaml')
"

# Install solver
echo "Installing CUDA-Q solver..."
python3 install_cuda_q_solver.py

echo ""
echo "Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review and customize cuda_q_config.yaml"
echo "2. Test the installation with: python3 -c 'from cuda_q_peaked_solver import create_cuda_q_solver; solver = create_cuda_q_solver(); print(\"CUDA-Q solver ready!\")'"
echo "3. Restart your quantum subnet miner"
echo ""
echo "For troubleshooting, check the logs and ensure your GPU drivers are up to date."
'''
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_path, 0o755)
        bt.logging.info(f"Installation script created: {output_path}")
    
    @staticmethod
    def create_docker_compose(output_path: str):
        """Create Docker Compose configuration for CUDA-Q"""
        docker_compose_content = '''version: '3.8'

services:
  quantum-miner-cuda-q:
    build:
      context: .
      dockerfile: Dockerfile.cuda-q
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=all
    volumes:
      - ./cuda_q_config.yaml:/app/cuda_q_config.yaml
      - ./logs:/app/logs
      - ./certificates:/app/certificates
    networks:
      - quantum-subnet
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  quantum-subnet:
    driver: bridge
'''
        
        dockerfile_content = '''FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install CUDA-Q
RUN pip3 install cuda-quantum

# Copy application code
COPY . .

# Install CUDA-Q solver
RUN python3 install_cuda_q_solver.py

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=all

# Run the miner
CMD ["python3", "-m", "neurons.miner"]
'''
        
        # Write Docker Compose file
        with open(output_path, 'w') as f:
            f.write(docker_compose_content)
        
        # Write Dockerfile
        dockerfile_path = Path(output_path).parent / "Dockerfile.cuda-q"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        bt.logging.info(f"Docker Compose configuration created: {output_path}")
        bt.logging.info(f"Dockerfile created: {dockerfile_path}")


def main():
    """Main function for configuration management CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA-Q Configuration Management")
    parser.add_argument("--create-config", help="Create default configuration file")
    parser.add_argument("--install-cuda-q", action="store_true", help="Install CUDA-Q")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU availability")
    parser.add_argument("--create-scripts", help="Create integration scripts in directory")
    
    args = parser.parse_args()
    
    if args.create_config:
        CudaQConfigManager.create_default_config(args.create_config)
        print(f"Default configuration created: {args.create_config}")
    
    if args.install_cuda_q:
        config = CudaQInstallationConfig()
        success = CudaQInstaller.install_cuda_q(config)
        if success:
            print("CUDA-Q installed successfully")
        else:
            print("CUDA-Q installation failed")
    
    if args.check_gpu:
        gpu_info = CudaQInstaller.check_gpu_availability()
        print("GPU Information:")
        for key, value in gpu_info.items():
            print(f"  {key}: {value}")
    
    if args.create_scripts:
        output_dir = Path(args.create_scripts)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        CudaQIntegrationHelper.create_solver_replacement_script(
            str(output_dir / "install_cuda_q_solver.py")
        )
        CudaQIntegrationHelper.create_installation_script(
            str(output_dir / "install_cuda_q.sh")
        )
        CudaQIntegrationHelper.create_docker_compose(
            str(output_dir / "docker-compose.cuda-q.yml")
        )
        
        print(f"Integration scripts created in: {output_dir}")


if __name__ == "__main__":
    main()

