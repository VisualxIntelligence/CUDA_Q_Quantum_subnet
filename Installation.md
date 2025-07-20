# CUDA-Q Installation Guide for Quantum Subnet

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Automated Installation](#automated-installation)
4. [Manual Installation](#manual-installation)
5. [Docker Installation](#docker-installation)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Post-Installation Validation](#post-installation-validation)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Ubuntu 22.04 LTS or compatible Linux distribution
- **Python**: Version 3.11 or higher
- **System Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 10GB free space for installation and temporary files
- **Network**: Stable internet connection for package downloads

### GPU Requirements

- **NVIDIA GPU**: GTX 1080 or newer, RTX series recommended
- **GPU Memory**: 8GB minimum, 24GB+ recommended for large circuits
- **CUDA Compute Capability**: 7.0 or higher
- **NVIDIA Driver**: Version 525.60.13 or newer
- **CUDA Toolkit**: Version 12.0 or higher

### Recommended Hardware Configurations

#### Single GPU Setup
- **GPU**: RTX 4090 (24GB) or A100 (40GB/80GB)
- **CPU**: Intel i7-12700K or AMD Ryzen 7 5800X
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 1TB NVMe SSD

#### Multi-GPU Setup
- **GPUs**: 2-4x RTX 4090 or 2-8x A100
- **CPU**: Intel i9-13900K or AMD Ryzen 9 7950X
- **RAM**: 64GB+ DDR4/DDR5
- **Storage**: 2TB+ NVMe SSD
- **PSU**: 1600W+ for multi-GPU configurations

#### Enterprise/Data Center Setup
- **GPUs**: 8x A100 80GB or H100
- **CPU**: Dual Xeon or EPYC processors
- **RAM**: 256GB+ ECC memory
- **Storage**: High-performance NVMe array
- **Network**: 100Gbps+ for multi-node setups

## Pre-Installation Checklist

### 1. Verify NVIDIA Driver Installation

```bash
nvidia-smi
```

Expected output should show your GPU(s) and driver version:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0  On |                  Off |
| 30%   45C    P8    25W / 450W |   1024MiB / 24564MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

### 2. Check CUDA Installation

```bash
nvcc --version
```

If CUDA is not installed, install it:
```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2
```

### 3. Verify Python Environment

```bash
python3 --version
pip3 --version
```

Ensure Python 3.11+ is installed:
```bash
# If needed, install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-dev
```

### 4. Check Available Disk Space

```bash
df -h
```

Ensure at least 10GB free space in your home directory and quantum subnet installation.

### 5. Verify Quantum Subnet Installation

```bash
cd /path/to/your/quantum/subnet
ls -la qbittensor/miner/solvers/
```

You should see `default_peaked_solver.py` in the solvers directory.

## Automated Installation

The automated installation script handles all dependencies and configuration automatically.

### 1. Download and Extract Package

```bash
cd /path/to/your/quantum/subnet
wget https://github.com/your-repo/cuda_q_quantum_subnet_package.tar.gz
tar -xzf cuda_q_quantum_subnet_package.tar.gz
cd cuda_q_quantum_subnet_package
```

### 2. Run Installation Script

```bash
chmod +x scripts/install_cuda_q.sh
./scripts/install_cuda_q.sh
```

The script will:
- Check system requirements
- Install CUDA-Q via pip
- Configure the enhanced solver
- Create default configuration files
- Validate the installation
- Backup original files

### 3. Monitor Installation Progress

The script provides detailed progress information:
```
CUDA-Q Installation Script for Quantum Subnet
==============================================
Checking prerequisites...
✓ Python 3.11 detected
✓ pip3 available
✓ NVIDIA drivers detected: RTX 4090 (24GB)

Installing CUDA-Q...
✓ CUDA-Q installed successfully

Creating default configuration...
✓ Configuration file created: cuda_q_config.yaml

Installing CUDA-Q solver...
✓ Original solver backed up
✓ CUDA-Q solver installed
✓ Imports updated

Installation completed successfully!
```

### 4. Verify Installation

```bash
python3 scripts/validate_installation.py
```

Expected output:
```
CUDA-Q Installation Validation
==============================
✓ CUDA-Q module import successful
✓ GPU detection working: 1 GPU(s) found
✓ CUDA-Q targets available: ['nvidia', 'nvidia-mgpu', 'nvidia-mqpu']
✓ Solver integration successful
✓ Configuration file valid

All checks passed! CUDA-Q is ready for use.
```

## Manual Installation

If the automated script fails or you prefer manual control, follow these detailed steps.

### 1. Install CUDA-Q

#### Option A: Install via pip (Recommended)

```bash
pip3 install cuda-quantum
```

#### Option B: Install via conda

```bash
conda install -c nvidia cuda-quantum
```

#### Option C: Install from source (Advanced)

```bash
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
python3 setup.py install
```

### 2. Verify CUDA-Q Installation

```bash
python3 -c "import cudaq; print(f'CUDA-Q version: {cudaq.__version__}')"
python3 -c "import cudaq; print(f'Available targets: {cudaq.get_targets()}')"
python3 -c "import cudaq; print(f'Number of GPUs: {cudaq.num_available_gpus()}')"
```

### 3. Backup Original Solver

```bash
cd /path/to/your/quantum/subnet
cp qbittensor/miner/solvers/default_peaked_solver.py qbittensor/miner/solvers/default_peaked_solver.py.backup
```

### 4. Install CUDA-Q Solver Files

```bash
cp cuda_q_quantum_subnet_package/src/cuda_q_peaked_solver.py qbittensor/miner/solvers/
cp cuda_q_quantum_subnet_package/src/cuda_q_config.py qbittensor/miner/solvers/
```

### 5. Update Miner Imports

Edit `qbittensor/miner/miner.py`:

**Before:**
```python
from .solvers.default_peaked_solver import DefaultPeakedSolver
```

**After:**
```python
from .solvers.cuda_q_peaked_solver import CudaQPeakedSolver as DefaultPeakedSolver
```

### 6. Create Configuration File

```bash
cp cuda_q_quantum_subnet_package/config/default_config.yaml cuda_q_config.yaml
```

### 7. Install Additional Dependencies

```bash
pip3 install numpy matplotlib pyyaml psutil
```

## Docker Installation

Docker provides an isolated environment with all dependencies pre-configured.

### 1. Install Docker and NVIDIA Container Runtime

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
```

### 2. Build CUDA-Q Docker Image

```bash
cd cuda_q_quantum_subnet_package/docker
docker build -f Dockerfile.cuda-q -t quantum-miner-cuda-q .
```

### 3. Run with Docker Compose

```bash
docker-compose up -d
```

### 4. Monitor Container

```bash
docker-compose logs -f quantum-miner-cuda-q
```

### 5. Access Container Shell

```bash
docker-compose exec quantum-miner-cuda-q bash
```

## Kubernetes Deployment

For production deployments with automatic scaling and management.

### 1. Install Kubernetes and NVIDIA GPU Operator

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/main/deployments/gpu-operator/gpu-operator.yaml
```

### 2. Deploy CUDA-Q Quantum Miner

```bash
kubectl apply -f cuda_q_quantum_subnet_package/docker/kubernetes.yaml
```

### 3. Monitor Deployment

```bash
kubectl get pods -l app=quantum-miner-cuda-q
kubectl logs -f deployment/quantum-miner-cuda-q
```

### 4. Scale Deployment

```bash
kubectl scale deployment quantum-miner-cuda-q --replicas=3
```

## Post-Installation Validation

### 1. Run Comprehensive Validation

```bash
python3 cuda_q_quantum_subnet_package/scripts/validate_installation.py --comprehensive
```

### 2. Performance Benchmark

```bash
cd cuda_q_quantum_subnet_package/tests
python3 cuda_q_benchmark.py --quick --plots
```

### 3. Integration Test

```bash
python3 cuda_q_quantum_subnet_package/tests/test_integration.py
```

### 4. Start Quantum Subnet Miner

```bash
cd /path/to/your/quantum/subnet
python3 -m neurons.miner
```

Monitor logs for CUDA-Q initialization:
```
INFO: CUDA-Q solver initialized with 1 GPUs, total memory: 24.0 GB
INFO: Selected backend: nvidia with strategy: single_gpu_statevector
INFO: Circuit solved successfully in 0.045s using single_gpu_statevector
```

## Troubleshooting

### Installation Issues

#### CUDA-Q Import Error

**Problem:**
```
ImportError: No module named 'cudaq'
```

**Solutions:**
1. Verify pip installation: `pip3 list | grep cuda-quantum`
2. Check Python path: `python3 -c "import sys; print(sys.path)"`
3. Reinstall CUDA-Q: `pip3 uninstall cuda-quantum && pip3 install cuda-quantum`
4. Use virtual environment: `python3 -m venv cuda_q_env && source cuda_q_env/bin/activate`

#### GPU Not Detected

**Problem:**
```
WARNING: No GPUs detected, falling back to CPU simulation
```

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Test CUDA-Q GPU detection: `python3 -c "import cudaq; print(cudaq.num_available_gpus())"`
4. Restart system after driver installation

#### Permission Errors

**Problem:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. Use `sudo` for system-wide installation
2. Install in user directory: `pip3 install --user cuda-quantum`
3. Check file permissions: `ls -la qbittensor/miner/solvers/`
4. Fix ownership: `sudo chown -R $USER:$USER /path/to/quantum/subnet`

### Runtime Issues

#### Memory Errors

**Problem:**
```
CUDA out of memory
```

**Solutions:**
1. Reduce circuit size or batch size
2. Enable memory pooling in configuration
3. Close other GPU applications
4. Restart to clear GPU memory

#### Performance Issues

**Problem:**
```
CUDA-Q slower than expected
```

**Solutions:**
1. Check GPU utilization: `nvidia-smi -l 1`
2. Enable performance profiling in configuration
3. Verify optimal backend selection
4. Update NVIDIA drivers

#### Configuration Errors

**Problem:**
```
Configuration file not found or invalid
```

**Solutions:**
1. Copy default configuration: `cp config/default_config.yaml cuda_q_config.yaml`
2. Validate YAML syntax: `python3 -c "import yaml; yaml.safe_load(open('cuda_q_config.yaml'))"`
3. Check file permissions and location

### Getting Additional Help

1. **Check Logs**: Review detailed logs in `/logs/` directory
2. **Run Diagnostics**: Use `python3 scripts/validate_installation.py --verbose`
3. **Hardware Check**: Run `python3 -c "from src.cuda_q_config import CudaQInstaller; print(CudaQInstaller.check_gpu_availability())"`
4. **Community Support**: Post issues in the quantum subnet repository with system information and error logs

### System Information for Support

When requesting help, include this information:

```bash
# System information
uname -a
lsb_release -a

# GPU information
nvidia-smi
nvcc --version

# Python environment
python3 --version
pip3 list | grep -E "(cuda|quantum|torch|numpy)"

# CUDA-Q status
python3 -c "import cudaq; print(f'CUDA-Q: {cudaq.__version__}'); print(f'GPUs: {cudaq.num_available_gpus()}'); print(f'Targets: {cudaq.get_targets()}')"
```

This comprehensive installation guide ensures successful deployment of CUDA-Q acceleration for your quantum subnet miner, enabling significant performance improvements and expanded computational capabilities.

