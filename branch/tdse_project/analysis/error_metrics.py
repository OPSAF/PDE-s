import numpy as np
from dataclasses import dataclass, field

@dataclass
class ErrorMetrics:
    """Compute and store error metrics between numerical and analytical solutions."""
    
    l2_error: float = 0.0          # L2 norm of error: sqrt(integral(|psi_num - psi_anal|² dx))
    max_error: float = 0.0         # Max pointwise error: max(|psi_num - psi_anal|)
    probability_error: float = 0.0 # |integral(|psi_num|²dx) - 1|
    
    @classmethod
    def compute(cls, psi_numerical: np.ndarray, psi_analytical: np.ndarray, 
                x: np.ndarray) -> 'ErrorMetrics':
        """Compute all error metrics."""
        dx = x[1] - x[0]
        
        # L2 error
        diff = psi_numerical - psi_analytical
        l2_err = float(np.sqrt(dx * np.sum(np.abs(diff)**2)))
        
        # Max error
        max_err = float(np.max(np.abs(diff)))
        
        # Probability conservation
        prob = float(dx * np.sum(np.abs(psi_numerical)**2))
        prob_err = abs(prob - 1.0)
        
        return cls(l2_error=l2_err, max_error=max_err, probability_error=prob_err)
    
    def to_dict(self) -> dict:
        return {
            'L2 Error': self.l2_error,
            'Max Error': self.max_error,
            'Probability Error': self.probability_error
        }
    
    def __str__(self) -> str:
        return (f"L2 Error: {self.l2_error:.6e}, "
                f"Max Error: {self.max_error:.6e}, "
                f"Prob Error: {self.probability_error:.6e}")
