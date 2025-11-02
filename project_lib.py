"""Imports for MOD300 Project 3"""
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

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


@dataclass(frozen=True, slots=True)
class Sphere:
    """Closed sphere defined by center (Å) and radius (Å)."""
    center: Tuple[float, float, float]
    radius: float

def random_sphere_in_box(
    box: "SimulationBox3D",
    r_min: float,
    r_max: float,
    rng: Optional[np.random.Generator] = None,
    *,
    fit_inside: bool = True,
    uniform_volume: bool = False,
) -> Sphere:
    """Sample a random sphere within the simulation box.

    Args:
        box: The SimulationBox3D to sample within.
        r_min: Minimum sphere radius (Å), must be > 0.
        r_max: Maximum sphere radius (Å), must be >= r_min.
        rng: Optional NumPy random Generator for reproducibility.
        fit_inside: If True, ensure the entire sphere is inside the box.
        uniform_volume: If True, sample radius so that sphere volume is uniform;
                        otherwise sample radius uniformly in [r_min, r_hi].

    Returns:
        Sphere(center=(cx, cy, cz), radius=r)

    Raises:
        ValueError: If parameters are invalid or the box is too small.
    """
    if r_min <= 0:
        raise ValueError("r_min must be > 0")
    if r_min > r_max:
        raise ValueError("r_min must be <= r_max")

    rng = rng or np.random.default_rng()

    # Limit the max radius so it can fit inside the box if requested.
    if fit_inside:
        lx = box.xmax - box.xmin
        ly = box.ymax - box.ymin
        lz = box.zmax - box.zmin
        r_limit = 0.5 * min(lx, ly, lz)
        r_hi = min(r_max, r_limit)
        if r_min > r_hi:
            raise ValueError(
                "Box too small for the requested radius range: "
                f"r_min={r_min:.3f}, max allowed within box={r_limit:.3f}"
            )
    else:
        r_hi = r_max

    # Sample radius
    if uniform_volume:
        # Make sphere volumes uniform: r^3 is uniform in [r_min^3, r_hi^3].
        u = rng.uniform(0.0, 1.0)
        r = ((r_min**3) + u * (r_hi**3 - r_min**3)) ** (1.0 / 3.0)
    else:
        r = rng.uniform(r_min, r_hi)

    # Sample center
    if fit_inside:
        cx = rng.uniform(box.xmin + r, box.xmax - r)
        cy = rng.uniform(box.ymin + r, box.ymax - r)
        cz = rng.uniform(box.zmin + r, box.zmax - r)
    else:
        cx = rng.uniform(box.xmin, box.xmax)
        cy = rng.uniform(box.ymin, box.ymax)
        cz = rng.uniform(box.zmin, box.zmax)

    return Sphere(center=(cx, cy, cz), radius=r)
