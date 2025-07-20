"""
CUDA-Q Benchmarking and Testing Suite
Comprehensive testing and performance evaluation for CUDA-Q integration
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    import cudaq
    CUDA_Q_AVAILABLE = True
except ImportError:
    CUDA_Q_AVAILABLE = False

from cuda_q_peaked_solver import CudaQPeakedSolver, CudaQConfig, HardwareDetector
from cuda_q_config import CudaQConfigManager

# For comparison with existing solver
try:
    from qbittensor.miner.solvers.default_peaked_solver import DefaultPeakedSolver
    QISKIT_SOLVER_AVAILABLE = True
except ImportError:
    QISKIT_SOLVER_AVAILABLE = False
    DefaultPeakedSolver = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    circuit_name: str
    num_qubits: int
    depth: int
    gate_count: int
    solver_type: str
    backend_used: str
    execution_time: float
    success: bool
    peak_bitstring: str
    memory_usage_gb: float
    speedup_factor: float = 1.0


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    timestamp: str
    hardware_info: Dict[str, Any]
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]


class CircuitGenerator:
    """Generates test circuits for benchmarking"""
    
    @staticmethod
    def generate_peaked_circuit(num_qubits: int, depth: int = None) -> str:
        """Generate a peaked circuit for testing"""
        if depth is None:
            depth = max(5, num_qubits // 2)
        
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];

"""
        
        # Add Hadamard gates to create superposition
        for i in range(min(3, num_qubits)):
            qasm += f"h q[{i}];\n"
        
        # Add entangling gates
        for layer in range(depth):
            for i in range(num_qubits - 1):
                if (layer + i) % 2 == 0:
                    qasm += f"cx q[{i}],q[{i+1}];\n"
        
        # Add some rotation gates for complexity
        for i in range(0, num_qubits, 2):
            angle = np.pi / (i + 1)
            qasm += f"ry({angle}) q[{i}];\n"
        
        # Add measurements
        qasm += f"measure q -> c;\n"
        
        return qasm
    
    @staticmethod
    def generate_ghz_circuit(num_qubits: int) -> str:
        """Generate a GHZ state circuit"""
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];

h q[0];
"""
        
        for i in range(1, num_qubits):
            qasm += f"cx q[0],q[{i}];\n"
        
        qasm += f"measure q -> c;\n"
        return qasm
    
    @staticmethod
    def generate_random_circuit(num_qubits: int, depth: int) -> str:
        """Generate a random quantum circuit"""
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];

"""
        
        gates = ['h', 'x', 'y', 'z', 's', 't']
        two_qubit_gates = ['cx', 'cz']
        
        np.random.seed(42)  # For reproducible results
        
        for layer in range(depth):
            # Single qubit gates
            for i in range(num_qubits):
                if np.random.random() < 0.3:
                    gate = np.random.choice(gates)
                    if gate in ['rx', 'ry', 'rz']:
                        angle = np.random.uniform(0, 2 * np.pi)
                        qasm += f"{gate}({angle}) q[{i}];\n"
                    else:
                        qasm += f"{gate} q[{i}];\n"
            
            # Two qubit gates
            for i in range(num_qubits - 1):
                if np.random.random() < 0.2:
                    gate = np.random.choice(two_qubit_gates)
                    j = (i + 1) % num_qubits
                    qasm += f"{gate} q[{i}],q[{j}];\n"
        
        qasm += f"measure q -> c;\n"
        return qasm


class PerformanceProfiler:
    """Profiles performance and resource usage"""
    
    @staticmethod
    def measure_memory_usage() -> float:
        """Measure current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb / 1024
        except ImportError:
            return 0.0
    
    @staticmethod
    def profile_execution(func, *args, **kwargs) -> Tuple[Any, float, float]:
        """Profile function execution time and memory usage"""
        start_memory = PerformanceProfiler.measure_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = PerformanceProfiler.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_delta = max(0, end_memory - start_memory)
        
        return result, execution_time, memory_delta


class CudaQBenchmark:
    """Main benchmarking class for CUDA-Q performance evaluation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = CudaQConfigManager.load_config(config_path)
        self.hardware_info = HardwareDetector.detect_hardware()
        self.results = []
        
        # Initialize solvers
        self.cuda_q_solver = CudaQPeakedSolver(self.config) if CUDA_Q_AVAILABLE else None
        self.qiskit_solver = DefaultPeakedSolver() if QISKIT_SOLVER_AVAILABLE else None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_single_benchmark(self, circuit_name: str, qasm: str, solver_type: str) -> BenchmarkResult:
        """Run a single benchmark test"""
        self.logger.info(f"Running benchmark: {circuit_name} with {solver_type}")
        
        # Analyze circuit
        from cuda_q_peaked_solver import CircuitAnalyzer
        circuit_info = CircuitAnalyzer.analyze_circuit(qasm)
        
        # Select solver
        if solver_type == "cuda_q" and self.cuda_q_solver:
            solver = self.cuda_q_solver
        elif solver_type == "qiskit" and self.qiskit_solver:
            solver = self.qiskit_solver
        else:
            raise ValueError(f"Solver {solver_type} not available")
        
        # Run benchmark
        try:
            result, execution_time, memory_usage = PerformanceProfiler.profile_execution(
                solver.solve, qasm
            )
            
            # Determine backend used
            backend_used = "unknown"
            if solver_type == "cuda_q" and hasattr(solver, 'backend_selector'):
                backend_config = solver.backend_selector.select_optimal_backend(
                    circuit_info, self.hardware_info
                )
                backend_used = f"{backend_config.backend_name}_{backend_config.execution_strategy}"
            elif solver_type == "qiskit":
                backend_used = "qiskit_aer"
            
            return BenchmarkResult(
                circuit_name=circuit_name,
                num_qubits=circuit_info.num_qubits,
                depth=circuit_info.depth,
                gate_count=circuit_info.gate_count,
                solver_type=solver_type,
                backend_used=backend_used,
                execution_time=execution_time,
                success=bool(result),
                peak_bitstring=result or "",
                memory_usage_gb=memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                circuit_name=circuit_name,
                num_qubits=circuit_info.num_qubits,
                depth=circuit_info.depth,
                gate_count=circuit_info.gate_count,
                solver_type=solver_type,
                backend_used="error",
                execution_time=0.0,
                success=False,
                peak_bitstring="",
                memory_usage_gb=0.0
            )
    
    def run_comparison_benchmark(self, circuit_name: str, qasm: str) -> List[BenchmarkResult]:
        """Run comparison benchmark between CUDA-Q and Qiskit"""
        results = []
        
        # Test with Qiskit first (baseline)
        if self.qiskit_solver:
            qiskit_result = self.run_single_benchmark(circuit_name, qasm, "qiskit")
            results.append(qiskit_result)
        
        # Test with CUDA-Q
        if self.cuda_q_solver:
            cuda_q_result = self.run_single_benchmark(circuit_name, qasm, "cuda_q")
            
            # Calculate speedup
            if len(results) > 0 and results[0].execution_time > 0:
                cuda_q_result.speedup_factor = results[0].execution_time / cuda_q_result.execution_time
            
            results.append(cuda_q_result)
        
        return results
    
    def run_scalability_benchmark(self) -> List[BenchmarkResult]:
        """Run scalability benchmark across different circuit sizes"""
        results = []
        
        # Test different circuit sizes
        qubit_counts = [12, 16, 20, 24, 28, 32, 36, 40]
        
        for num_qubits in qubit_counts:
            circuit_name = f"peaked_{num_qubits}q"
            qasm = CircuitGenerator.generate_peaked_circuit(num_qubits)
            
            # Run comparison for this circuit size
            circuit_results = self.run_comparison_benchmark(circuit_name, qasm)
            results.extend(circuit_results)
            
            # Stop if circuits become too large for available memory
            if any(r.execution_time > 300 for r in circuit_results):  # 5 minute timeout
                self.logger.warning(f"Stopping scalability test at {num_qubits} qubits due to timeout")
                break
        
        return results
    
    def run_circuit_type_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark across different circuit types"""
        results = []
        
        circuit_types = [
            ("peaked_20q", CircuitGenerator.generate_peaked_circuit(20)),
            ("ghz_20q", CircuitGenerator.generate_ghz_circuit(20)),
            ("random_20q", CircuitGenerator.generate_random_circuit(20, 10)),
            ("peaked_30q", CircuitGenerator.generate_peaked_circuit(30)),
            ("ghz_30q", CircuitGenerator.generate_ghz_circuit(30)),
            ("random_30q", CircuitGenerator.generate_random_circuit(30, 15)),
        ]
        
        for circuit_name, qasm in circuit_types:
            circuit_results = self.run_comparison_benchmark(circuit_name, qasm)
            results.extend(circuit_results)
        
        return results
    
    def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run the complete benchmark suite"""
        self.logger.info("Starting full benchmark suite")
        
        all_results = []
        
        # Run scalability benchmark
        self.logger.info("Running scalability benchmark...")
        scalability_results = self.run_scalability_benchmark()
        all_results.extend(scalability_results)
        
        # Run circuit type benchmark
        self.logger.info("Running circuit type benchmark...")
        circuit_type_results = self.run_circuit_type_benchmark()
        all_results.extend(circuit_type_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_results)
        
        return BenchmarkSuite(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware_info=asdict(self.hardware_info),
            results=all_results,
            summary_stats=summary_stats
        )
    
    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark results"""
        cuda_q_results = [r for r in results if r.solver_type == "cuda_q" and r.success]
        qiskit_results = [r for r in results if r.solver_type == "qiskit" and r.success]
        
        stats = {
            "total_tests": len(results),
            "cuda_q_tests": len(cuda_q_results),
            "qiskit_tests": len(qiskit_results),
            "cuda_q_success_rate": len(cuda_q_results) / max(1, len([r for r in results if r.solver_type == "cuda_q"])),
            "qiskit_success_rate": len(qiskit_results) / max(1, len([r for r in results if r.solver_type == "qiskit"])),
        }
        
        if cuda_q_results:
            stats["cuda_q_avg_time"] = np.mean([r.execution_time for r in cuda_q_results])
            stats["cuda_q_median_time"] = np.median([r.execution_time for r in cuda_q_results])
            stats["cuda_q_avg_memory"] = np.mean([r.memory_usage_gb for r in cuda_q_results])
        
        if qiskit_results:
            stats["qiskit_avg_time"] = np.mean([r.execution_time for r in qiskit_results])
            stats["qiskit_median_time"] = np.median([r.execution_time for r in qiskit_results])
            stats["qiskit_avg_memory"] = np.mean([r.memory_usage_gb for r in qiskit_results])
        
        # Calculate speedup statistics
        speedups = [r.speedup_factor for r in cuda_q_results if r.speedup_factor > 0]
        if speedups:
            stats["avg_speedup"] = np.mean(speedups)
            stats["median_speedup"] = np.median(speedups)
            stats["max_speedup"] = np.max(speedups)
            stats["min_speedup"] = np.min(speedups)
        
        return stats
    
    def save_results(self, benchmark_suite: BenchmarkSuite, output_path: str):
        """Save benchmark results to file"""
        # Convert to serializable format
        data = {
            "timestamp": benchmark_suite.timestamp,
            "hardware_info": benchmark_suite.hardware_info,
            "summary_stats": benchmark_suite.summary_stats,
            "results": [asdict(r) for r in benchmark_suite.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_path}")
    
    def generate_performance_plots(self, benchmark_suite: BenchmarkSuite, output_dir: str):
        """Generate performance visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Execution time comparison by circuit size
        self._plot_execution_time_comparison(benchmark_suite, output_path / "execution_time_comparison.png")
        
        # Speedup factor plot
        self._plot_speedup_factors(benchmark_suite, output_path / "speedup_factors.png")
        
        # Memory usage comparison
        self._plot_memory_usage(benchmark_suite, output_path / "memory_usage.png")
        
        # Success rate comparison
        self._plot_success_rates(benchmark_suite, output_path / "success_rates.png")
    
    def _plot_execution_time_comparison(self, benchmark_suite: BenchmarkSuite, output_path: Path):
        """Plot execution time comparison"""
        cuda_q_results = [r for r in benchmark_suite.results if r.solver_type == "cuda_q" and r.success]
        qiskit_results = [r for r in benchmark_suite.results if r.solver_type == "qiskit" and r.success]
        
        if not cuda_q_results or not qiskit_results:
            return
        
        # Group by number of qubits
        cuda_q_by_qubits = {}
        qiskit_by_qubits = {}
        
        for r in cuda_q_results:
            if r.num_qubits not in cuda_q_by_qubits:
                cuda_q_by_qubits[r.num_qubits] = []
            cuda_q_by_qubits[r.num_qubits].append(r.execution_time)
        
        for r in qiskit_results:
            if r.num_qubits not in qiskit_by_qubits:
                qiskit_by_qubits[r.num_qubits] = []
            qiskit_by_qubits[r.num_qubits].append(r.execution_time)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        qubits = sorted(set(cuda_q_by_qubits.keys()) & set(qiskit_by_qubits.keys()))
        cuda_q_times = [np.mean(cuda_q_by_qubits[q]) for q in qubits]
        qiskit_times = [np.mean(qiskit_by_qubits[q]) for q in qubits]
        
        plt.plot(qubits, cuda_q_times, 'o-', label='CUDA-Q', linewidth=2, markersize=8)
        plt.plot(qubits, qiskit_times, 's-', label='Qiskit Aer', linewidth=2, markersize=8)
        
        plt.xlabel('Number of Qubits')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison: CUDA-Q vs Qiskit Aer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speedup_factors(self, benchmark_suite: BenchmarkSuite, output_path: Path):
        """Plot speedup factors"""
        cuda_q_results = [r for r in benchmark_suite.results if r.solver_type == "cuda_q" and r.speedup_factor > 0]
        
        if not cuda_q_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        qubits = [r.num_qubits for r in cuda_q_results]
        speedups = [r.speedup_factor for r in cuda_q_results]
        
        plt.scatter(qubits, speedups, alpha=0.7, s=100)
        plt.xlabel('Number of Qubits')
        plt.ylabel('Speedup Factor (CUDA-Q vs Qiskit)')
        plt.title('CUDA-Q Speedup Factors by Circuit Size')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add horizontal line at speedup = 1
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, benchmark_suite: BenchmarkSuite, output_path: Path):
        """Plot memory usage comparison"""
        cuda_q_results = [r for r in benchmark_suite.results if r.solver_type == "cuda_q" and r.success]
        qiskit_results = [r for r in benchmark_suite.results if r.solver_type == "qiskit" and r.success]
        
        if not cuda_q_results or not qiskit_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        cuda_q_qubits = [r.num_qubits for r in cuda_q_results]
        cuda_q_memory = [r.memory_usage_gb for r in cuda_q_results]
        qiskit_qubits = [r.num_qubits for r in qiskit_results]
        qiskit_memory = [r.memory_usage_gb for r in qiskit_results]
        
        plt.scatter(cuda_q_qubits, cuda_q_memory, alpha=0.7, label='CUDA-Q', s=100)
        plt.scatter(qiskit_qubits, qiskit_memory, alpha=0.7, label='Qiskit Aer', s=100)
        
        plt.xlabel('Number of Qubits')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, benchmark_suite: BenchmarkSuite, output_path: Path):
        """Plot success rates by circuit size"""
        # Group results by solver type and qubit count
        success_data = {}
        
        for result in benchmark_suite.results:
            key = (result.solver_type, result.num_qubits)
            if key not in success_data:
                success_data[key] = {"total": 0, "success": 0}
            
            success_data[key]["total"] += 1
            if result.success:
                success_data[key]["success"] += 1
        
        # Calculate success rates
        cuda_q_rates = {}
        qiskit_rates = {}
        
        for (solver_type, num_qubits), data in success_data.items():
            rate = data["success"] / data["total"]
            if solver_type == "cuda_q":
                cuda_q_rates[num_qubits] = rate
            elif solver_type == "qiskit":
                qiskit_rates[num_qubits] = rate
        
        if not cuda_q_rates or not qiskit_rates:
            return
        
        plt.figure(figsize=(12, 8))
        
        qubits = sorted(set(cuda_q_rates.keys()) & set(qiskit_rates.keys()))
        cuda_q_success = [cuda_q_rates[q] for q in qubits]
        qiskit_success = [qiskit_rates[q] for q in qubits]
        
        plt.plot(qubits, cuda_q_success, 'o-', label='CUDA-Q', linewidth=2, markersize=8)
        plt.plot(qubits, qiskit_success, 's-', label='Qiskit Aer', linewidth=2, markersize=8)
        
        plt.xlabel('Number of Qubits')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison by Circuit Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for running benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA-Q Benchmark Suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer tests)")
    parser.add_argument("--plots", action="store_true", help="Generate performance plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = CudaQBenchmark(args.config)
    
    if args.quick:
        # Quick benchmark - just a few test cases
        results = []
        test_circuits = [
            ("peaked_16q", CircuitGenerator.generate_peaked_circuit(16)),
            ("ghz_20q", CircuitGenerator.generate_ghz_circuit(20)),
            ("random_24q", CircuitGenerator.generate_random_circuit(24, 10)),
        ]
        
        for circuit_name, qasm in test_circuits:
            circuit_results = benchmark.run_comparison_benchmark(circuit_name, qasm)
            results.extend(circuit_results)
        
        summary_stats = benchmark._calculate_summary_stats(results)
        benchmark_suite = BenchmarkSuite(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware_info=asdict(benchmark.hardware_info),
            results=results,
            summary_stats=summary_stats
        )
    else:
        # Full benchmark suite
        benchmark_suite = benchmark.run_full_benchmark_suite()
    
    # Save results
    results_file = output_path / "benchmark_results.json"
    benchmark.save_results(benchmark_suite, str(results_file))
    
    # Generate plots if requested
    if args.plots:
        plots_dir = output_path / "plots"
        benchmark.generate_performance_plots(benchmark_suite, str(plots_dir))
        print(f"Performance plots saved to {plots_dir}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)
    for key, value in benchmark_suite.summary_stats.items():
        print(f"{key}: {value}")
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()

