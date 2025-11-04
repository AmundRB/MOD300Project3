"""Imports for MOD300 Project 3"""
from typing import List, Optional, Tuple, Iterable, Sequence
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
        return abs(self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

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
) -> tuple[float, float]:
    """Backward-compatible wrapper for a single sphere (old Task 4 API)."""
    return estimate_fraction_inside_spheres(box, [sphere], n_samples, rng)

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
    pts = box.random_point(n=n_samples, rng=rng)  

    # Count how many are inside the sphere
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

    # Binomial std error for p, then scale to pi
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

# Task 6
def generate_random_spheres(
    box: "SimulationBox3D",
    n: int,
    r_min: float,
    r_max: float,
    rng: Optional[np.random.Generator] = None,
) -> List["Sphere"]:
    """Generate n random spheres in the box.

    By default, spheres are allowed to overlap. Each sphere fully fits in the box
    if fit_inside=True. Radii are uniform in [r_min, r_max] unless uniform_volume=True.

    Args:
        box: Simulation box.
        n: number of spheres to generate.
        r_min, r_max: radius range (Å).
        rng: optional NumPy Generator for reproducibility.
        uniform_volume: if True, sample radii uniformly in volume (r^3).
        fit_inside: if True, ensure spheres are entirely inside the box.

    Returns:
        List of Sphere objects.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    rng = rng or np.random.default_rng()

    spheres: List[Sphere] = []
    for _ in range(n):
        s = random_sphere_in_box(
            box=box,
            r_min=r_min,
            r_max=r_max,
            rng=rng
        )
        spheres.append(s)
    return spheres

# Task 8
# Units in pm
def get_atom_radius(atom: str) -> int:
    """Return the radius of an atom in Å."""
    radius: dict[str, int] = {
    "H": 120,
    "C": 170,
    "N": 155,
    "O": 152,
    "P": 180,
    }
    assert atom in radius, f"Atom {atom} not recognized."
    
    return radius[atom] * 0.01  # Convert pm to Å

def create_sphere_from_dna_file(
    filename: str,
) -> list[Sphere]:
    """Create spheres from a DNA file.

    Args:
        filename: Path to the DNA file.

    Returns:
        List of Sphere objects.
    """
    spheres = []
    with open(filename, "r") as f:
        for line in f:
            lineParts = line.strip().split()
            if len(lineParts) != 4:
                continue
            atomtype = lineParts[0]
            x = float(lineParts[1])
            y = float(lineParts[2])
            z = float(lineParts[3])
            radius = get_atom_radius(atomtype)

            spheres.append(Sphere(center=(x, y, z), radius=radius))
    return spheres

def bounding_box_for_spheres(
    spheres: Iterable[Sphere]
) -> SimulationBox3D:
    """Tight axis-aligned box that contains all spheres (+ optional margin in Å)."""
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")

    for s in spheres:
        cx, cy, cz = s.center
        r = s.radius
        xmin = min(xmin, cx - r)
        ymin = min(ymin, cy - r)
        zmin = min(zmin, cz - r)
        xmax = max(xmax, cx + r)
        ymax = max(ymax, cy + r)
        zmax = max(zmax, cz + r)

    return SimulationBox3D(
        x=(xmin - 1, xmax + 1),
        y=(ymin - 1, ymax + 1),
        z=(zmin - 1, zmax + 1),
    )

def estimate_fraction_inside_spheres(
    box: "SimulationBox3D",
    spheres: Sequence["Sphere"],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Estimate fraction of box volume covered by the union of many spheres.

    Args:
        box: SimulationBox3D defining the sampling region.
        spheres: sequence of Sphere objects (e.g. all DNA atoms).
        n_samples: number of random points.
        rng: optional NumPy Generator for reproducibility.

    Returns:
        (fraction, stderr) where
        - fraction ≈ (# points inside any sphere) / n_samples
        - stderr   ≈ sqrt(p * (1 - p) / n_samples)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not spheres:
        raise ValueError("spheres must be non-empty")

    rng = rng or np.random.default_rng()

    
    pts = box.random_point(n=n_samples, rng=rng)
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    # inside_any[i] is True if point i is inside at least one sphere
    inside_any = np.zeros(n_samples, dtype=bool)

    for s in spheres:
        cx, cy, cz = s.center
        r2 = s.radius * s.radius
        dx = xs - cx
        dy = ys - cy
        dz = zs - cz
        inside_this = (dx * dx + dy * dy + dz * dz) < r2
        inside_any |= inside_this 

    count_inside = int(np.count_nonzero(inside_any))
    fraction = count_inside / float(n_samples)
    stderr = math.sqrt(max(fraction * (1.0 - fraction), 0.0) / n_samples)
    return fraction, stderr

# Topic 2 Task 1

# Function for handling walkers trying to step outside the box
def _reflect_on_boundary(
        val: float,
        lo: float,
        hi: float
        ) -> float:
    """
    If a walker(point) tries to move outside the box, reflect in like bouncing of a wall
    args:
        val (float): The value to reflect.
        lo (float): Lower bound of the box.
        hi (float): Upper bound of the box.

    returns:
        float: The value to reflect.
    """
    while val < lo or val > hi:
        if val < lo:
            val = lo + (lo - val)
        elif val > hi:
            val = hi - (val - hi)
    return val

# Function for handling "one step"
def _sample_step(
        step_sigma: float,
        rng: np.random.Generator
) -> np.ndarray:
    """One random 3D step"""
    return rng.normal(0.0, step_sigma, size=3)

def _simulate_single_walker(
        start: np.ndarray,
        num_steps: int,
        step_sigma: float,
        box
) -> np.ndarray:
    """
    One random 3D walker with reflecting boundary
    args:
        start (np.ndarray): Initial position, shape (3,).
        num_steps (int): Number of steps to simulate (>= 0).
        step_sigma (float): Standard deviation of the Gaussian step per axis.
        box: SimulationBox3D-like object with (xmin, xmax, ymin, ymax, zmin, zmax).

    returns:
        np.ndarray: Trajectory of shape (num_steps+1, 3), including start.
    """
    # initializing variables
    rng = np.random.default_rng()
    traj = np.empty((num_steps + 1, 3), dtype=float)
    pos = np.array(start, dtype=float)
    traj[0] = pos

    # Simulating walking for one walker and storing in array traj
    for t in range(1, num_steps + 1):
        pos = pos + _sample_step(step_sigma, rng)
        pos[0] = _reflect_on_boundary(pos[0], box.xmin, box.xmax)
        pos[1] = _reflect_on_boundary(pos[1], box.ymin, box.ymax)
        pos[2] = _reflect_on_boundary(pos[2], box.zmin, box.zmax)
        traj[t] = pos
    
    return traj

# Function for generating a set of random walkers in 3D starting from random points
def random_walkers_3D(
        box,
        num_walkers: int,
        num_steps: int,
        step_sigma: float,
) -> np.ndarray:
    """
    Generates a set of random walkers in 3D from random starting points.
    args:
        box: SimulationBox3D-like object with (xmin, xmax, ymin, ymax, zmin, zmax).
        num_walkers (int): Number of walkers to simulate (> 0).
        num_steps (int): Number of steps per walker (>= 0).
        step_sigma (float): Standard deviation of the Gaussian step per axis (> 0).

    returns:
        np.ndarray: Array of trajectories with shape (num_walkers, num_steps+1, 3).
    """
    # Input validation
    if num_walkers <= 0 or num_steps < 0:
        raise ValueError("num_walkers must be > 0 and num_steps >= 0")
    if step_sigma <= 0:
        raise ValueError("step_sigma must be > 0")
    
    # Variable initialization
    rng = np.random.default_rng()
    starts = box.random_point(n=num_walkers, rng=rng)
    all_traj = np.empty((num_walkers, num_steps + 1, 3), dtype=float)

    # Simulate walking for all walkers with num_steps
    for w in range(num_walkers):
        all_traj[w] = _simulate_single_walker(
            start=starts[w],
            num_steps=num_steps,
            step_sigma=step_sigma,
            box=box,
        )
    return all_traj