"""Imports for MOD300 Project 3"""
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math

# Task 0
class SimulationBox3D:
    """Axis-aligned 3D simulation box (Å)."""

    def __init__(
            self,
            x: Tuple[float, float],
            y: Tuple[float, float],
            z: Tuple[float, float],
            ) -> None:
        self.xmin, self.xmax = x
        self.ymin, self.ymax = y
        self.zmin, self.zmax = z
        if not (self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax):
            raise ValueError("Invalid bounds: require xmin<xmax, ymin<ymax, zmin<zmax")

    def __str__(self) -> str:
        """return a string with dimensions of SimulationBox3D as:
        x: (xmin, xmax),
        y: (ymin, ymax)
        z: (zmin, zmax)"""
        return (
            "SimulationBox dimensions:\n"
            f"  x: ({self.xmin}, {self.xmax})\n"
            f"  y: ({self.ymin}, {self.ymax})\n"
            f"  z: ({self.zmin}, {self.zmax})"
        )

    def volume(self) -> float:
        """Calculate the volume of the simulation box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    def surface_area(self) -> float:
        """Calculates and returns the surface area of the simulation box."""
        return 2 * ((self.xmax - self.xmin) * (self.ymax - self.ymin) +
                     (self.ymax - self.ymin) * (self.zmax - self.zmin) +
                     (self.zmax - self.zmin) * (self.xmax - self.xmin))

    def is_point_inside(self, x, y, z) -> bool:
        """Check if a point (x, y, z) is inside the simulation box."""
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)
    
    # Task 1
    def random_point(
    self,
    n: int | None = None,
    rng: Optional[np.random.Generator] = None
    ) -> tuple[float, float, float]:
        """Uniformly sample point(s) inside the box.

        Args:
            n: If None return one point as a tuple (x, y, z).
            If an int return an (n, 3) NumPy array of points.
            rng: Optional numpy random Generator for reproducibility.
                If None, uses np.random.default_rng().

        Returns:
            (x, y, z) tuple if n is None, else an (n, 3) ndarray.
        """
        rng = rng or np.random.default_rng()

        if n is None:
            x = rng.uniform(self.xmin, self.xmax)
            y = rng.uniform(self.ymin, self.ymax)
            z = rng.uniform(self.zmin, self.zmax)
            return (x, y, z)

        xs = rng.uniform(self.xmin, self.xmax, size=n)
        ys = rng.uniform(self.ymin, self.ymax, size=n)
        zs = rng.uniform(self.zmin, self.zmax, size=n)
        return np.column_stack((xs, ys, zs))

# Task 2
@dataclass(frozen=True, slots=True)
class Sphere:
    """Closed sphere defined by center (Å) and radius (Å)."""
    center: Tuple[float, float, float]
    radius: float
    # Task 3
    def is_point_in_sphere(self, point: tuple[float, float, float]) -> bool:
        """
        Return True if a point is stricly inside a sphere. 
        Boundary of sphere counts as outside 
        """
        d_x = point[0] - self.center[0]
        d_y = point[1] - self.center[1]
        d_z = point[2] - self.center[2]
        return (d_x * d_x + d_y * d_y + d_z * d_z) < (self.radius * self.radius)
    def volume(self) -> float:
        """Calculate the volume of the sphere."""
        return (4.0 / 3.0) * math.pi * (self.radius ** 3)

def _max_radius_that_fits(box) -> float:
    """Compute max radius that fits inside the box"""
    len_x = box.xmax - box.xmin
    len_y = box.ymax - box.ymin
    len_z = box.zmax - box.zmin
    return 0.5 * min(len_x, len_y, len_z)

def _sample_radius(r_min, r_max_allowed, rng):
    """picks a random radius within allowed radius range uniformly"""
    return rng.uniform(r_min, r_max_allowed)

def _sample_center_inside(box, r, rng) -> tuple:
    """picks a random center point for the sphere inside the box"""
    cx = rng.uniform(box.xmin + r, box.xmax - r)
    cy = rng.uniform(box.ymin + r, box.ymax - r)
    cz = rng.uniform(box.zmin + r, box.zmax - r)
    return (cx, cy, cz)

def random_sphere_in_box(
    box: "SimulationBox3D",
    r_min: float,
    r_max: float,
    rng: Optional[np.random.Generator] = None,
) -> Sphere:
    """Sample a random sphere within the simulation box.

    Args:
        box: The SimulationBox3D to sample within.
        r_min: Minimum sphere radius (Å), must be > 0.
        r_max: Maximum sphere radius (Å), must be >= r_min.
        rng: Optional NumPy random Generator for reproducibility.

    Returns:
        Sphere(center=(cx, cy, cz), radius=r)

    Raises:
        ValueError: If parameters are invalid or the box is too small.
    """
    # Input Validation
    if r_min <= 0:
        raise ValueError("r_min must be > 0")
    if r_min > r_max:
        raise ValueError("r_min must be <= r_max")

    rng = rng or np.random.default_rng()

    #Ensure the sphere fits inside the box
    r_limit = _max_radius_that_fits(box)
    r_max_allowed = min(r_max, r_limit)
    if r_min > r_max_allowed:
        raise ValueError(
            f"Box too small: r_min={r_min:.3f}, max allowed={r_max_allowed:.3f}"
        )

    r = _sample_radius(r_min, r_max_allowed, rng)
    center = _sample_center_inside(box, r, rng)

    return Sphere(center=center, radius=r)

# Task 4

def estimate_fraction_inside_sphere(
    box: "SimulationBox3D",
    sphere: "Sphere",
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
):
    """Estimate fraction of box volume covered by a single sphere via Monte Carlo.

    Returns:
        fraction: float in [0,1]
        stderr:   binomial standard error ≈ sqrt(p*(1-p)/n_samples)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = rng or np.random.default_rng()

    # Sample N points uniformly in the box (vectorized)
    pts = box.random_point(n=n_samples, rng=rng)  # shape (N, 3)

    # Vectorized point-in-sphere: ||p - c||^2 < r^2
    cx, cy, cz = sphere.center
    r2 = sphere.radius * sphere.radius
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dz = pts[:, 2] - cz
    inside = (dx * dx + dy * dy + dz * dz) < r2

    count_inside = int(np.count_nonzero(inside))
    fraction = count_inside / float(n_samples)

    # Binomial standard error (good MC uncertainty proxy)
    stderr = math.sqrt(max(fraction * (1.0 - fraction), 0.0) / n_samples)
    return fraction, stderr

#Task 5

def estimate_pi(
    box: "SimulationBox3D",
    sphere: "Sphere",
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
):
    """Estimate pi using uniform samples in an existing box that fully contains the sphere.
    Returns (pi_hat, stderr_pi)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    rng = rng or np.random.default_rng()

    # Sample points in the given box
    pts = box.random_point(n=n_samples, rng=rng)  # (N, 3)

    # Count how many are inside the sphere (strictly inside per your definition)
    cx, cy, cz = sphere.center
    r = sphere.radius
    r2 = r * r
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dz = pts[:, 2] - cz
    inside = (dx*dx + dy*dy + dz*dz) < r2
    count_inside = int(np.count_nonzero(inside))

    # Fraction and conversion to pi
    p_hat = count_inside / float(n_samples)
    V_box = box.volume()
    K = (3.0 * V_box) / (4.0 * r**3)  # scale factor such that pi_hat = K * p_hat
    pi_hat = K * p_hat

    # Binomial SE for p, then scale to pi
    stderr_p = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n_samples)
    stderr_pi = K * stderr_p
    return pi_hat, stderr_pi


def run_pi_experiment(
    box: "SimulationBox3D",
    sphere: "Sphere",
    Ns: list[int],
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[float], list[float]]:
    """Convenience helper: compute pi-hat and error for a list of sample sizes."""
    rng = rng or np.random.default_rng()
    ests, errs = [], []
    for N in Ns:
        pi_hat, err = estimate_pi(box, sphere, n_samples=N, rng=rng)
        ests.append(pi_hat)
        errs.append(err)
    return Ns, ests, errs