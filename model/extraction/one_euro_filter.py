"""
One Euro Filter — adaptive low-pass filter for noisy real-time signals.

Reduces jitter in stationary signals while minimizing lag during fast
movements. Based on the algorithm by Casiez, Roussel, and Vogel (CHI 2012).

Reference: https://gery.casiez.net/1euro/
"""
import numpy as np


class OneEuroFilter:
    """Speed-adaptive low-pass filter for a single scalar signal."""

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """
        Initialize the filter with a first sample.

        Args:
            t0: Initial timestamp in seconds.
            x0: Initial signal value.
            min_cutoff: Minimum cutoff frequency (Hz). Lower means more
                smoothing when stationary.
            beta: Speed coefficient. Higher means less lag during fast movement.
            d_cutoff: Cutoff frequency for the derivative low-pass filter (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        """
        Filter a new sample.

        Args:
            t: Current timestamp in seconds.
            x: Current noisy value.

        Returns:
            The filtered value.
        """
        t = float(t)
        x = float(x)
        te = t - self.t_prev

        if te <= 0.0:
            return x

        ad = self._alpha(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = ad * dx + (1 - ad) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def _alpha(self, te: float, cutoff: float) -> float:
        """
        Compute the smoothing factor from time elapsed and cutoff frequency.

        Args:
            te: Time elapsed since last sample in seconds.
            cutoff: Current cutoff frequency in Hz.

        Returns:
            Smoothing factor alpha in range (0, 1].
        """
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
