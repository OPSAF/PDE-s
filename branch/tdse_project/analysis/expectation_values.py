import numpy as np
from dataclasses import dataclass

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    dx = x[1] - x[0]
    return float(np.real(dx * (np.sum(y) - 0.5 * (y[0] + y[-1]))))

@dataclass 
class ExpectationValues:
    """Quantum mechanical expectation values."""
    x_exp: float = 0.0     # <x>
    p_exp: float = 0.0     # <p> (computed via momentum-space or finite difference)
    x2_exp: float = 0.0    # <x²>
    p2_exp: float = 0.0    # <p²>
    energy: float = 0.0    # <H> = <p²>/2m + <V>
    
    @classmethod
    def compute(cls, psi: np.ndarray, x: np.ndarray, V: np.ndarray = None,
                hbar: float = 1.0, m: float = 1.0) -> 'ExpectationValues':
        """Compute expectation values for wavefunction psi."""
        dx = x[1] - x[0]
        prob = np.abs(psi)**2
        
        # <x>
        x_exp = float(trapz(x * prob, x))
        
        # <x²>
        x2_exp = float(trapz(x**2 * prob, x))
        
        # <p> via momentum representation: p = -i*hbar*d/dx
        # Use central difference for derivative: dpsi/dx[i] ≈ (psi[i+1] - psi[i-1])/(2*dx)
        dpsi_dx = np.zeros_like(psi, dtype=complex)
        dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2 * dx)
        p_op = -1j * hbar * dpsi_dx  # p operator applied to psi
        # <p> = integral(psi* . p_op . psi) dx
        p_exp = float(np.real(trapz(np.conj(psi) * p_op, x)))
        
        # <p²> via second derivative: d²psi/dx²[i] ≈ (psi[i+1] - 2*psi[i] + psi[i-1])/dx²
        d2psi_dx2 = np.zeros_like(psi, dtype=complex)
        d2psi_dx2[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
        # p² = -hbar² * d²/dx²
        p2_op = -hbar**2 * d2psi_dx2
        p2_exp = float(np.real(trapz(np.conj(psi) * p2_op, x)))
        
        # Energy <H> = <p²>/(2m) + <V>
        if V is not None:
            v_exp = float(trapz(V * prob, x))
            energy = p2_exp / (2 * m) + v_exp
        else:
            v_exp = 0.0
            energy = p2_exp / (2 * m)
        
        return cls(x_exp=x_exp, p_exp=p_exp, x2_exp=x2_exp, 
                   p2_exp=p2_exp, energy=energy)
    
    def to_dict(self) -> dict:
        return {
            '<x>': self.x_exp,
            '<p>': self.p_exp,
            '<x²>': self.x2_exp,
            '<p²>': self.p2_exp,
            '<E>': self.energy
        }
