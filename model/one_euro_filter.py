import numpy as np
import math

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Initialize the One Euro Filter.
        
        Args:
            t0: Initial timestamp.
            x0: Initial value.
            min_cutoff: Minimum cutoff frequency (Hz). Lower means more filtering when stationary.
            beta: Speed coefficient. Higher means less lag during movement.
            d_cutoff: Cutoff frequency for the derivative (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """
        Filter a value.
        
        Args:
            t: Current timestamp.
            x: Current value to filter.
            
        Returns:
            The filtered value.
        """
        t = float(t)
        x = float(x)
        te = t - self.t_prev
        
        # Avoid division by zero if timestamps are identical
        if te <= 0.0: 
            return x

        # Calculate derivation
        ad = self._alpha(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        
        # Calculate cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        # Update state
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def _alpha(self, te, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
