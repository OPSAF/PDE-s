import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings

@dataclass
class ConvergenceResult:
    """Results from a convergence study."""
    parameter_name: str           # e.g., 'Nx' or 'dt'
    parameter_values: List[float] # Grid sizes or time steps tested
    errors: List[float]           # Corresponding errors (L2)
    observed_order: Optional[float] = None  # Estimated convergence order
    
    def estimate_order(self) -> float:
        """Estimate convergence order from last two points."""
        if len(self.errors) >= 2 and len(self.parameter_values) >= 2:
            n = len(self.errors)
            # order = log(e_n / e_{n-1}) / log(h_n / h_{n-1})
            # For spatial: h ~ 1/Nx, so ratio = Nx_{n-1}/Nx_n
            # For temporal: directly use dt ratio
            e_ratio = self.errors[-2] / self.errors[-1] if self.errors[-1] > 0 else float('inf')
            p_ratio = self.parameter_values[-2] / self.parameter_values[-1]
            if p_ratio > 1 and e_ratio > 1:
                self.observed_order = np.log(e_ratio) / np.log(p_ratio)
            else:
                self.observed_order = 0.0
        return self.observed_order if self.observed_order else 0.0

class ConvergenceStudy:
    """Perform spatial and temporal convergence studies."""
    
    def __init__(self, output_dir: str = ''):
        self.output_dir = output_dir
        self.spatial_results: Dict[str, ConvergenceResult] = {}
        self.temporal_results: Dict[str, ConvergenceResult] = {}
    
    def spatial_convergence(self, model, solver_class, nx_list: list = [128, 256, 512, 1024],
                           x_range: tuple = (-20.0, 20.0), dt: float = 0.001,
                           total_time: float = 1.0, verbose: bool = True) -> ConvergenceResult:
        """
        Study spatial convergence by varying Nx.
        
        Returns ConvergenceResult with errors vs Nx.
        """
        # Local trapz to avoid relative import issues
        def _trapz(y, x):
            dx = x[1] - x[0]
            return float(np.real(dx * (np.sum(y) - 0.5 * (y[0] + y[-1]))))

        errors = []
        for nx in nx_list:
            x = np.linspace(x_range[0], x_range[1], nx)
            V = model.potential(x)
            psi0 = model.initial_wavefunction(x)
            
            solver = solver_class(x)
            psi = psi0.copy()
            
            num_steps = int(total_time / dt)
            for _ in range(num_steps):
                psi = solver.propagate(psi, V, dt)
            
            psi_exact = model.analytical_solution(x, total_time)
            
            dx = x[1] - x[0]
            diff = psi - psi_exact
            l2_err = float(np.sqrt(dx * np.sum(np.abs(diff)**2)))
            errors.append(l2_err)
            
            if verbose:
                print(f"  Nx={nx:5d}: L2 error = {l2_err:.6e}")
        
        result = ConvergenceResult(
            parameter_name='Nx',
            parameter_values=[float(n) for n in nx_list],
            errors=errors
        )
        result.estimate_order()
        
        if verbose:
            print(f"  Observed spatial convergence order: {result.observed_order:.2f}")
        
        return result
    
    def temporal_convergence(self, model, solver_class, nx: int = 512,
                            x_range: tuple = (-20.0, 20.0),
                            dt_list: list = [0.01, 0.005, 0.001],
                            total_time: float = 1.0, verbose: bool = True) -> ConvergenceResult:
        """
        Study temporal convergence by varying dt.
        
        Returns ConvergenceResult with errors vs dt.
        """
        errors = []
        x = np.linspace(x_range[0], x_range[1], nx)
        V = model.potential(x)
        psi_exact = model.analytical_solution(x, total_time)
        
        for dt in dt_list:
            psi0 = model.initial_wavefunction(x)
            solver = solver_class(x)
            psi = psi0.copy()
            
            num_steps = int(total_time / dt)
            for _ in range(num_steps):
                psi = solver.propagate(psi, V, dt)
            
            dx = x[1] - x[0]
            diff = psi - psi_exact
            l2_err = float(np.sqrt(dx * np.sum(np.abs(diff)**2)))
            errors.append(l2_err)
            
            if verbose:
                print(f"  dt={dt:.4f}: L2 error = {l2_err:.6e}")
        
        result = ConvergenceResult(
            parameter_name='dt',
            parameter_values=dt_list,
            errors=errors
        )
        result.estimate_order()
        
        if verbose:
            print(f"  Observed temporal convergence order: {result.observed_order:.2f}")
        
        return result
    
    def run_full_study(self, model, solver_class, label: str = '',
                      nx_list: list = None, dt_list: list = None) -> Dict[str, Any]:
        """Run both spatial and temporal convergence studies."""
        if nx_list is None:
            nx_list = [128, 256, 512, 1024]
        if dt_list is None:
            dt_list = [0.01, 0.005, 0.001]
        
        print(f"\n{'='*60}")
        print(f"Convergence Study: {label}")
        print(f"{'='*60}")
        
        print("\n[Spatial Convergence]")
        spatial = self.spatial_convergence(model, solver_class, nx_list=nx_list)
        self.spatial_results[label] = spatial
        
        print("\n[Temporal Convergence]")  
        temporal = self.temporal_convergence(model, solver_class, dt_list=dt_list)
        self.temporal_results[label] = temporal
        
        return {
            'spatial': spatial,
            'temporal': temporal,
            'spatial_order': spatial.observed_order,
            'temporal_order': temporal.observed_order
        }
