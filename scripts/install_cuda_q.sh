#!/bin/bash
# CUDA-Q Installation Script for Quantum Subnet
# Automated installation and configuration of CUDA-Q multi-GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for safety reasons."
   exit 1
fi

echo -e "${BLUE}"
echo "=============================================="
echo "CUDA-Q Installation Script for Quantum Subnet"
echo "=============================================="
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
log_info "Checking prerequisites..."

# Check Python
if ! command_exists python3; then
    log_error "Python 3 is required but not installed."
    log_info "Install with: sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_success "Python $PYTHON_VERSION detected"

# Check pip
if ! command_exists pip3; then
    log_error "pip3 is required but not installed."
    log_info "Install with: sudo apt install python3-pip"
    exit 1
fi
log_success "pip3 available"

# Check NVIDIA drivers
if ! command_exists nvidia-smi; then
    log_warning "nvidia-smi not found. CUDA-Q will fall back to CPU simulation."
    log_info "For GPU acceleration, install NVIDIA drivers and CUDA toolkit."
    GPU_AVAILABLE=false
else
    log_success "NVIDIA drivers detected"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    log_info "GPU: $GPU_INFO"
    GPU_AVAILABLE=true
fi

# Check CUDA (optional)
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_success "CUDA $CUDA_VERSION detected"
else
    log_warning "CUDA toolkit not found. Some features may be limited."
fi

# Check available disk space
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
REQUIRED_SPACE=10485760  # 10GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    log_error "Insufficient disk space. Required: 10GB, Available: $(($AVAILABLE_SPACE/1024/1024))GB"
    exit 1
fi
log_success "Sufficient disk space available"

# Check if we're in the quantum subnet directory
if [ ! -d "qbittensor" ]; then
    log_error "Not in quantum subnet directory. Please run this script from your quantum subnet root."
    log_info "Expected directory structure: qbittensor/miner/solvers/"
    exit 1
fi
log_success "Quantum subnet directory structure detected"

# Install CUDA-Q
log_info "Installing CUDA-Q..."
if pip3 install cuda-quantum; then
    log_success "CUDA-Q installed successfully"
else
    log_error "Failed to install CUDA-Q"
    log_info "Try manual installation: pip3 install --user cuda-quantum"
    exit 1
fi

# Verify CUDA-Q installation
log_info "Verifying CUDA-Q installation..."
if python3 -c "import cudaq; print(f'CUDA-Q version: {cudaq.__version__}'); print(f'Available targets: {cudaq.get_targets()}')"; then
    log_success "CUDA-Q verification successful"
else
    log_error "CUDA-Q verification failed"
    exit 1
fi

# Install additional dependencies
log_info "Installing additional dependencies..."
pip3 install numpy matplotlib pyyaml psutil

# Create backup of original solver
log_info "Creating backup of original solver..."
if [ -f "qbittensor/miner/solvers/default_peaked_solver.py" ]; then
    cp qbittensor/miner/solvers/default_peaked_solver.py qbittensor/miner/solvers/default_peaked_solver.py.backup
    log_success "Original solver backed up"
else
    log_warning "Original solver not found at expected location"
fi

# Copy CUDA-Q solver files
log_info "Installing CUDA-Q solver..."
cp cuda_q_quantum_subnet_package/src/cuda_q_peaked_solver.py qbittensor/miner/solvers/
cp cuda_q_quantum_subnet_package/src/cuda_q_config.py qbittensor/miner/solvers/
log_success "CUDA-Q solver files installed"

# Update miner imports
log_info "Updating miner imports..."
MINER_FILE="qbittensor/miner/miner.py"
if [ -f "$MINER_FILE" ]; then
    # Create backup
    cp "$MINER_FILE" "${MINER_FILE}.backup"
    
    # Update import
    sed -i 's/from \.solvers\.default_peaked_solver import DefaultPeakedSolver/from .solvers.cuda_q_peaked_solver import CudaQPeakedSolver as DefaultPeakedSolver/' "$MINER_FILE"
    log_success "Miner imports updated"
else
    log_warning "Miner file not found. Manual import update required."
fi

# Create default configuration
log_info "Creating default configuration..."
if [ "$GPU_AVAILABLE" = true ]; then
    cp cuda_q_quantum_subnet_package/config/default_config.yaml cuda_q_config.yaml
else
    # Create CPU-only configuration
    cp cuda_q_quantum_subnet_package/config/default_config.yaml cuda_q_config.yaml
    sed -i 's/nvidia_enabled: true/nvidia_enabled: false/' cuda_q_config.yaml
    sed -i 's/mgpu_enabled: true/mgpu_enabled: false/' cuda_q_config.yaml
    sed -i 's/mqpu_enabled: true/mqpu_enabled: false/' cuda_q_config.yaml
fi
log_success "Configuration file created: cuda_q_config.yaml"

# Create logs directory
mkdir -p logs
log_success "Logs directory created"

# Run validation
log_info "Running installation validation..."
if python3 cuda_q_quantum_subnet_package/scripts/validate_installation.py; then
    log_success "Installation validation passed"
else
    log_warning "Installation validation had issues. Check the output above."
fi

# Final success message
echo ""
echo -e "${GREEN}"
echo "=============================================="
echo "Installation completed successfully!"
echo "=============================================="
echo -e "${NC}"

echo ""
echo "Next steps:"
echo "1. Review and customize cuda_q_config.yaml if needed"
echo "2. Test the installation:"
echo "   python3 -c \"from qbittensor.miner.solvers.cuda_q_peaked_solver import create_cuda_q_solver; solver = create_cuda_q_solver(); print('CUDA-Q solver ready!')\""
echo "3. Run performance benchmark (optional):"
echo "   cd cuda_q_quantum_subnet_package/tests && python3 cuda_q_benchmark.py --quick"
echo "4. Restart your quantum subnet miner:"
echo "   python3 -m neurons.miner"
echo ""

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}GPU acceleration is enabled and ready!${NC}"
else
    echo -e "${YELLOW}Running in CPU mode. Install NVIDIA drivers for GPU acceleration.${NC}"
fi

echo ""
echo "For troubleshooting, check:"
echo "- Installation logs above"
echo "- cuda_q_quantum_subnet_package/INSTALLATION.md"
echo "- cuda_q_quantum_subnet_package/docs/TROUBLESHOOTING.md"
echo ""
echo "Happy quantum computing! 🚀"

