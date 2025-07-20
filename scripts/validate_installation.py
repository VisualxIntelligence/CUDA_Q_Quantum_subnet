#!/usr/bin/env python3
"""
CUDA-Q Installation Validation Script
Comprehensive validation of CUDA-Q integration with quantum subnet
"""

import sys
import os
import importlib
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message: str):
    print(f"{Colors.GREEN}[✓]{Colors.NC} {message}")

def log_warning(message: str):
    print(f"{Colors.YELLOW}[⚠]{Colors.NC} {message}")

def log_error(message: str):
    print(f"{Colors.RED}[✗]{Colors.NC} {message}")

class ValidationResult:
    def __init__(self, name: str, success: bool, message: str, details: Dict[str, Any] = None):
        self.name = name
        self.success = success
        self.message = message
        self.details = details or {}

class CudaQValidator:
    def __init__(self, comprehensive: bool = False):
        self.comprehensive = comprehensive
        self.results: List[ValidationResult] = []
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print(f"{Colors.BLUE}")
        print("=" * 50)
        print("CUDA-Q Installation Validation")
        print("=" * 50)
        print(f"{Colors.NC}")
        
        # Basic validations
        self.validate_python_environment()
        self.validate_cuda_q_import()
        self.validate_gpu_detection()
        self.validate_cuda_q_targets()
        self.validate_solver_integration()
        self.validate_configuration()
        
        if self.comprehensive:
            self.validate_performance()
            self.validate_circuit_execution()
            self.validate_backend_selection()
        
        # Print summary
        self.print_summary()
        
        # Return overall success
        return all(result.success for result in self.results)
    
    def validate_python_environment(self):
        """Validate Python environment and dependencies"""
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if sys.version_info >= (3, 11):
                self.results.append(ValidationResult(
                    "Python Version", True, f"Python {python_version} (✓ >= 3.11)"
                ))
            else:
                self.results.append(ValidationResult(
                    "Python Version", False, f"Python {python_version} (✗ < 3.11 required)"
                ))
            
            # Check required packages
            required_packages = ['numpy', 'yaml', 'matplotlib']
            for package in required_packages:
                try:
                    importlib.import_module(package)
                    self.results.append(ValidationResult(
                        f"Package {package}", True, f"{package} available"
                    ))
                except ImportError:
                    self.results.append(ValidationResult(
                        f"Package {package}", False, f"{package} not found"
                    ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Python Environment", False, f"Error checking Python environment: {e}"
            ))
    
    def validate_cuda_q_import(self):
        """Validate CUDA-Q import and basic functionality"""
        try:
            import cudaq
            
            # Check version
            version = getattr(cudaq, '__version__', 'unknown')
            self.results.append(ValidationResult(
                "CUDA-Q Import", True, f"CUDA-Q {version} imported successfully"
            ))
            
            # Test basic kernel creation
            kernel = cudaq.make_kernel()
            qubit = kernel.qalloc()
            kernel.h(qubit)
            kernel.mz(qubit)
            
            self.results.append(ValidationResult(
                "CUDA-Q Kernel Creation", True, "Basic kernel creation successful"
            ))
            
        except ImportError as e:
            self.results.append(ValidationResult(
                "CUDA-Q Import", False, f"Failed to import CUDA-Q: {e}"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                "CUDA-Q Basic Test", False, f"CUDA-Q basic test failed: {e}"
            ))
    
    def validate_gpu_detection(self):
        """Validate GPU detection and CUDA availability"""
        try:
            import cudaq
            
            # Check GPU count
            num_gpus = cudaq.num_available_gpus()
            if num_gpus > 0:
                self.results.append(ValidationResult(
                    "GPU Detection", True, f"{num_gpus} GPU(s) detected",
                    {"num_gpus": num_gpus}
                ))
            else:
                self.results.append(ValidationResult(
                    "GPU Detection", False, "No GPUs detected - will use CPU simulation"
                ))
            
            # Check NVIDIA-SMI if available
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                       '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')
                    self.results.append(ValidationResult(
                        "NVIDIA Drivers", True, f"NVIDIA drivers working, {len(gpu_info)} GPU(s) found"
                    ))
                else:
                    self.results.append(ValidationResult(
                        "NVIDIA Drivers", False, "nvidia-smi failed"
                    ))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.results.append(ValidationResult(
                    "NVIDIA Drivers", False, "nvidia-smi not available"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "GPU Detection", False, f"Error checking GPU availability: {e}"
            ))
    
    def validate_cuda_q_targets(self):
        """Validate available CUDA-Q targets"""
        try:
            import cudaq
            
            targets = cudaq.get_targets()
            if targets:
                target_list = ', '.join(targets)
                self.results.append(ValidationResult(
                    "CUDA-Q Targets", True, f"Available targets: {target_list}",
                    {"targets": targets}
                ))
                
                # Check for specific targets
                expected_targets = ['nvidia', 'nvidia-mgpu', 'nvidia-mqpu']
                for target in expected_targets:
                    if target in targets:
                        self.results.append(ValidationResult(
                            f"Target {target}", True, f"{target} target available"
                        ))
                    else:
                        self.results.append(ValidationResult(
                            f"Target {target}", False, f"{target} target not available"
                        ))
            else:
                self.results.append(ValidationResult(
                    "CUDA-Q Targets", False, "No CUDA-Q targets available"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "CUDA-Q Targets", False, f"Error checking CUDA-Q targets: {e}"
            ))
    
    def validate_solver_integration(self):
        """Validate solver integration with quantum subnet"""
        try:
            # Check if solver files exist
            solver_path = Path("qbittensor/miner/solvers/cuda_q_peaked_solver.py")
            config_path = Path("qbittensor/miner/solvers/cuda_q_config.py")
            
            if solver_path.exists():
                self.results.append(ValidationResult(
                    "Solver File", True, "CUDA-Q solver file found"
                ))
            else:
                self.results.append(ValidationResult(
                    "Solver File", False, "CUDA-Q solver file not found"
                ))
            
            if config_path.exists():
                self.results.append(ValidationResult(
                    "Config File", True, "CUDA-Q config file found"
                ))
            else:
                self.results.append(ValidationResult(
                    "Config File", False, "CUDA-Q config file not found"
                ))
            
            # Try to import the solver
            sys.path.insert(0, str(Path("qbittensor/miner/solvers").absolute()))
            try:
                from cuda_q_peaked_solver import CudaQPeakedSolver, create_cuda_q_solver
                
                self.results.append(ValidationResult(
                    "Solver Import", True, "CUDA-Q solver imported successfully"
                ))
                
                # Try to create solver instance
                solver = create_cuda_q_solver()
                self.results.append(ValidationResult(
                    "Solver Creation", True, "CUDA-Q solver instance created"
                ))
                
            except ImportError as e:
                self.results.append(ValidationResult(
                    "Solver Import", False, f"Failed to import CUDA-Q solver: {e}"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    "Solver Creation", False, f"Failed to create solver: {e}"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Solver Integration", False, f"Error validating solver integration: {e}"
            ))
    
    def validate_configuration(self):
        """Validate configuration file"""
        try:
            import yaml
            
            config_file = Path("cuda_q_config.yaml")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check required configuration keys
                required_keys = ['enabled', 'nvidia_enabled', 'mgpu_enabled']
                missing_keys = [key for key in required_keys if key not in config]
                
                if not missing_keys:
                    self.results.append(ValidationResult(
                        "Configuration File", True, "Configuration file valid"
                    ))
                else:
                    self.results.append(ValidationResult(
                        "Configuration File", False, f"Missing keys: {missing_keys}"
                    ))
            else:
                self.results.append(ValidationResult(
                    "Configuration File", False, "cuda_q_config.yaml not found"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Configuration File", False, f"Error validating configuration: {e}"
            ))
    
    def validate_performance(self):
        """Validate performance capabilities (comprehensive mode only)"""
        try:
            sys.path.insert(0, str(Path("qbittensor/miner/solvers").absolute()))
            from cuda_q_peaked_solver import create_cuda_q_solver
            
            solver = create_cuda_q_solver()
            
            # Test with a simple circuit
            test_qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0],q[1];
            cx q[1],q[2];
            measure q -> c;
            """
            
            import time
            start_time = time.time()
            result = solver.solve(test_qasm)
            execution_time = time.time() - start_time
            
            if result:
                self.results.append(ValidationResult(
                    "Performance Test", True, 
                    f"Test circuit solved in {execution_time:.3f}s, result: {result}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Performance Test", False, "Test circuit failed to solve"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Performance Test", False, f"Performance test failed: {e}"
            ))
    
    def validate_circuit_execution(self):
        """Validate circuit execution capabilities"""
        try:
            import cudaq
            
            # Test simple circuit execution
            kernel = cudaq.make_kernel()
            qubit = kernel.qalloc()
            kernel.h(qubit)
            kernel.mz(qubit)
            
            # Try to sample
            result = cudaq.sample(kernel, shots_count=100)
            
            if result:
                self.results.append(ValidationResult(
                    "Circuit Execution", True, "Basic circuit execution successful"
                ))
            else:
                self.results.append(ValidationResult(
                    "Circuit Execution", False, "Circuit execution returned no results"
                ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Circuit Execution", False, f"Circuit execution failed: {e}"
            ))
    
    def validate_backend_selection(self):
        """Validate backend selection logic"""
        try:
            sys.path.insert(0, str(Path("qbittensor/miner/solvers").absolute()))
            from cuda_q_peaked_solver import BackendSelector, CudaQConfig, HardwareDetector
            
            config = CudaQConfig()
            selector = BackendSelector(config)
            hardware_info = HardwareDetector.detect_hardware()
            
            # Test backend selection for different circuit sizes
            from cuda_q_peaked_solver import CircuitAnalyzer
            
            test_qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[12];
            creg c[12];
            h q[0];
            measure q -> c;
            """
            
            circuit_info = CircuitAnalyzer.analyze_circuit(test_qasm)
            backend_config = selector.select_optimal_backend(circuit_info, hardware_info)
            
            self.results.append(ValidationResult(
                "Backend Selection", True, 
                f"Backend selection working: {backend_config.backend_name}"
            ))
        
        except Exception as e:
            self.results.append(ValidationResult(
                "Backend Selection", False, f"Backend selection failed: {e}"
            ))
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("Validation Summary")
        print("=" * 50)
        
        success_count = sum(1 for result in self.results if result.success)
        total_count = len(self.results)
        
        for result in self.results:
            if result.success:
                log_success(f"{result.name}: {result.message}")
            else:
                log_error(f"{result.name}: {result.message}")
        
        print("\n" + "-" * 50)
        if success_count == total_count:
            log_success(f"All {total_count} checks passed! CUDA-Q is ready for use.")
        else:
            failed_count = total_count - success_count
            log_warning(f"{success_count}/{total_count} checks passed. {failed_count} issues found.")
            
            print("\nRecommendations:")
            for result in self.results:
                if not result.success:
                    if "import" in result.message.lower():
                        print("  - Install missing packages: pip3 install cuda-quantum numpy yaml matplotlib")
                    elif "gpu" in result.message.lower():
                        print("  - Install NVIDIA drivers and CUDA toolkit for GPU acceleration")
                    elif "file" in result.message.lower():
                        print("  - Run the installation script: ./scripts/install_cuda_q.sh")
                    elif "config" in result.message.lower():
                        print("  - Copy default configuration: cp config/default_config.yaml cuda_q_config.yaml")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CUDA-Q installation")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive validation including performance tests")
    parser.add_argument("--json", help="Output results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = CudaQValidator(comprehensive=args.comprehensive)
    success = validator.run_all_validations()
    
    # Save JSON output if requested
    if args.json:
        results_data = {
            "success": success,
            "total_checks": len(validator.results),
            "passed_checks": sum(1 for r in validator.results if r.success),
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "message": r.message,
                    "details": r.details
                }
                for r in validator.results
            ]
        }
        
        with open(args.json, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        log_info(f"Results saved to {args.json}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

