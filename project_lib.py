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
        box,
        rng: np.random.Generator | None = None,
        steps: np.ndarray | None = None,
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
    rng = rng or np.random.default_rng()
    traj = np.empty((num_steps + 1, 3), dtype=float)
    pos = np.array(start, dtype=float)
    traj[0] = pos

    # Simulating walking for one walker and storing in array traj
    for t in range(1, num_steps + 1):
        step = steps[t-1] if steps is not None else _sample_step(step_sigma, rng)
        pos = pos + step
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
        starts: np.ndarray | None = None,
        steps: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
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
    rng = rng or np.random.default_rng()
    if starts is None:
        starts = box.random_point(n=num_walkers, rng=rng)
    all_traj = np.empty((num_walkers, num_steps + 1, 3), dtype=float)

    # Simulate walking for all walkers with num_steps
    for w in range(num_walkers):
        all_traj[w] = _simulate_single_walker(
            start=starts[w],
            num_steps=num_steps,
            step_sigma=step_sigma,
            box=box,
            steps=None if steps is None else steps[w]
        )
    return all_traj

# Topic 2 Task 2

def _reflect_on_boundary_vectorized(
        coords: np.ndarray,
        lo: float,
        hi: float
) -> np.ndarray:
    """
    If a walker tries to move outside the box, reflect in like bouncing of a wall: Vectorized
    args:
        coords: np.ndarray of positions
        lo: lower bound of box
        hi: upper bound of box
    
    returns:
        np.ndarray with all values reflected into the box
    """
    # loop reflects until every entry lies inside the box
    while True:
        under = coords < lo
        over  = coords > hi

        if not (under.any() or over.any()):
            break

        # mirror the out-of-bounds values back in
        coords[under] = 2.0 * lo - coords[under]
        coords[over]  = 2.0 * hi - coords[over]

    return coords
    
def random_walkers_3D_fast(
        box,
        num_walkers: int,
        num_steps: int,
        step_sigma: float,
        starts: np.ndarray | None = None,
        steps: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Vectorized 3D random walkers with reflecting boundaries. Generates all steps at once

    args:
        box: SimulationBox3D-like with (xmin, xmax, ymin, ymax, zmin, zmax).
        num_walkers: number of walkers (> 0).
        num_steps: steps per walker (>= 0).
        step_sigma: Gaussian step std-dev per axis (> 0).
        rng: optional NumPy Generator; if None, a new default_rng() is used.
    
    returns:
        np.ndarray of shape (num_walkers, num_steps+1, 3): full trajectories.
    """
    # Input validation
    if num_walkers <= 0 or num_steps < 0:
        raise ValueError("num_walkers must be > 0 and num_steps >= 0")
    if step_sigma <= 0:
        raise ValueError("step_sigma must be > 0")
    
    # Variable initialization
    rng = rng or np.random.default_rng()
    if starts is None:
        starts = box.random_point(n=num_walkers, rng=rng)

    # use provbided steps or make all random steps in one go
    if steps is None:
        steps = rng.normal(0.0, step_sigma, size=(num_walkers, num_steps, 3))


    # Creating trajectories with starting position
    traj = np.empty((num_walkers, num_steps + 1, 3), dtype=float)
    traj[:, 0, :] = starts
    pos = starts.copy()

    # creating the movement and using reflect on boundary function to reflect coordinates into the box boundary
    for t in range(num_steps):
        pos = pos + steps[:, t, :]
        pos[:, 0] = _reflect_on_boundary_vectorized(pos[:, 0], box.xmin, box.xmax)
        pos[:, 1] = _reflect_on_boundary_vectorized(pos[:, 1], box.ymin, box.ymax)
        pos[:, 2] = _reflect_on_boundary_vectorized(pos[:, 2], box.zmin, box.zmax)
        traj[:, t + 1, :] = pos

    return traj

def assert_traj_shape_and_bounds(traj: np.ndarray, box: "SimulationBox3D",
                                 num_walkers: int, num_steps: int) -> None:
    """
    description:
        Validate trajectory array shape and that all coordinates is inside the box.
    args:
        traj (np.ndarray): Trajectories, shape (num_walkers, num_steps+1, 3).
        box (SimulationBox3D): 3D simulation box
        num_walkers (int): walker count.
        num_steps (int): step count per walker.
    returns:
        None. Raises AssertionError on mismatch.
    """
    assert traj.shape == (num_walkers, num_steps + 1, 3), "Unexpected trajectory shape"
    xmin, xmax = box.xmin, box.xmax
    ymin, ymax = box.ymin, box.ymax
    zmin, zmax = box.zmin, box.zmax
    xs, ys, zs = traj[..., 0], traj[..., 1], traj[..., 2]
    assert np.all((xs >= xmin) & (xs <= xmax)), "x out of bounds"
    assert np.all((ys >= ymin) & (ys <= ymax)), "y out of bounds"
    assert np.all((zs >= zmin) & (zs <= zmax)), "z out of bounds"
    print("Assert checks on shape and boundary ok!")


def run_walkers_slow_fast(box: "SimulationBox3D",
                          num_walkers: int,
                          num_steps: int,
                          step_sigma: float,
                          steps: np.ndarray | None = None,
                          seed: int = 42,
                          ) -> dict[str, object]:
    """
    description:
        Run the loop-based (slow) and vectorized (fast) random walkers with shared starts & RNG seed,
        time both, and perform basic assertions.
    args:
        box (SimulationBox3D): 3D simulation box.
        num_walkers (int): Number of walkers.
        num_steps (int): Steps per walker.
        step_sigma (float): Std dev of Gaussian step per axis.
        seed (int): RNG seed used for both runs and shared starts.
    returns:
        dict with keys:
            "traj_slow" (np.ndarray), "traj_fast" (np.ndarray),
            "time_slow" (float), "time_fast" (float),
            "starts" (np.ndarray)
    """
    from time import perf_counter
    rng_starts = np.random.default_rng(seed)
    starts = box.random_point(n=num_walkers, rng=rng_starts)

    # run and time the slow random walkers function, run assertion checks.
    rng_slow = np.random.default_rng(seed)
    t0 = perf_counter()
    traj_slow = random_walkers_3D(
        box=box,
        num_walkers=num_walkers,
        num_steps=num_steps,
        step_sigma=step_sigma,
        starts=starts,
        steps=steps,
        rng=rng_slow,
    )
    t1 = perf_counter()
    time_slow = t1 - t0
    assert_traj_shape_and_bounds(traj_slow, box, num_walkers, num_steps)

    # Run and time the fast random walkers function, run assertion checks
    rng_fast = np.random.default_rng(seed)
    t0 = perf_counter()
    traj_fast = random_walkers_3D_fast(
        box=box,
        num_walkers=num_walkers,
        num_steps=num_steps,
        step_sigma=step_sigma,
        starts=starts,
        rng=rng_fast,
        steps=steps
    )
    t1 = perf_counter()
    time_fast = t1 - t0
    assert_traj_shape_and_bounds(traj_fast, box, num_walkers, num_steps)

    return {
        "traj_slow": traj_slow,
        "traj_fast": traj_fast,
        "time_slow": time_slow,
        "time_fast": time_fast,
        "starts": starts,
    }


def plot_walkers_subset(traj_slow: np.ndarray,
                        traj_fast: np.ndarray,
                        box: "SimulationBox3D",
                        subset: int = 5,
                        title: str | None = None) -> None:
    """
    description:
        Plot the same subset of walkers for slow vs fast side-by-side.
    args:
        traj_slow (np.ndarray): Slow trajectories, shape (W, T+1, 3).
        traj_fast (np.ndarray): Fast trajectories, shape (W, T+1, 3).
        box (SimulationBox3D): 3D simulation box.
        subset (int): Number of walkers to show from the start.
        title (str|None): Optional suptitle for the figure.
    returns:
        None. Displays a matplotlib figure.
    """
    import matplotlib.pyplot as plt

    W = min(subset, traj_slow.shape[0], traj_fast.shape[0])
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for w in range(W):
        ax1.plot(traj_slow[w, :, 0], traj_slow[w, :, 1], traj_slow[w, :, 2], alpha=0.85)
    ax1.set_title("Regular random walkers (slow)")
    ax1.set_xlim(box.xmin, box.xmax); ax1.set_ylim(box.ymin, box.ymax); ax1.set_zlim(box.zmin, box.zmax)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for w in range(W):
        ax2.plot(traj_fast[w, :, 0], traj_fast[w, :, 1], traj_fast[w, :, 2], alpha=0.85)
    ax2.set_title("Fast vectorized walkers")
    ax2.set_xlim(box.xmin, box.xmax); ax2.set_ylim(box.ymin, box.ymax); ax2.set_zlim(box.zmin, box.zmax)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Topic 2 Task 5

def is_point_inside_sphere(
        point: np.ndarray,
        spheres: Sequence["Sphere"],
) -> bool:
    """
    Returns true if a 3D point lies inside any of the spheres in the box.
    """
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    for s in spheres:
        cx, cy, cz = s.center
        dx = x - cx
        dy = y - cy
        dz = z - cz
        if dx*dx + dy*dy + dz*dz < (s.radius * s.radius):
            return True
    return False

def sample_start_outside_spheres(
        box: "SimulationBox3D",
        spheres: Sequence["Sphere"],
        rng: np.random.Generator,
) -> np.ndarray:
    """
    finds a single point inside the box that is not inside a sphere
    """
    while True:
        # use existing uniform random point function
        x, y, z = box.random_point(rng=rng)
        p = np.array([x, y, z], dtype=float)
        # accept only if it lies in the accessible region of the box
        if not is_point_inside_sphere(p, spheres):
            return p
        
def sample_starts_outside_spheres(
        box: "SimulationBox3D",
        spheres: Sequence["Sphere"],
        num_walkers: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Finds valid starting positions(not inside any sphere) for all the walkers
    """
    if num_walkers <= 0:
        raise ValueError("num_walkers must be > 0")
    starts = np.empty((num_walkers, 3), dtype=float)
    for w in range(num_walkers):
        starts[w] = sample_start_outside_spheres(box, spheres, rng)
    return starts

def random_walkers_avoiding_spheres(
        box: "SimulationBox3D",
        spheres: Sequence["Sphere"],
        num_walkers: int,
        num_steps: int,
        step_sigma: float,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random walkers with:
      - reflecting box boundaries (reuses _reflect_on_boundary)
      - "hard" spherical obstacles: if a proposed step lands inside a sphere,
        the step is rejected and the walker stays where it was.

    Args:
        box: 3D simulation box.
        spheres: obstacle spheres (e.g. DNA atoms).
        num_walkers: number of independent walkers (>0).
        num_steps: number of steps per walker (>=0).
        step_sigma: std dev of Gaussian step per axis (>0).
        rng: optional NumPy Generator.

    Returns:
        traj: array of shape (num_walkers, num_steps+1, 3),
              containing full trajectories for all walkers.
    """
    # Input Validation
    if num_walkers <= 0 or num_steps < 0:
        raise ValueError("num_walkers must be > 0 and num_steps >= 0")
    if step_sigma <= 0:
        raise ValueError("step_sigma must be > 0")

    rng = rng or np.random.default_rng()

    # Start all walkers in valid starting positions inside the box
    starts = sample_starts_outside_spheres(box, spheres, num_walkers, rng)

    # Build trajectory arrays and simulate walkers
    traj = np.empty((num_walkers, num_steps + 1, 3), dtype=float)
    for w in range(num_walkers):
        pos = np.array(starts[w], dtype=float)
        traj[w, 0, :] = pos
        for t in range(1, num_steps + 1):
            # try a random step(reusing _sample_step function)
            step = _sample_step(step_sigma, rng)
            test_step = pos + step

            # reflect on box boundaries
            test_step[0] = _reflect_on_boundary(test_step[0], box.xmin, box.xmax)
            test_step[1] = _reflect_on_boundary(test_step[1], box.ymin, box.ymax)
            test_step[2] = _reflect_on_boundary(test_step[2], box.zmin, box.zmax)

            # Check if step lands inside a sphere. If we land inside we reject the step.
            if is_point_inside_sphere(test_step, spheres):
                # no update of pos
                traj[w, t, :] = pos
            else:
                # accept step
                pos = test_step
                traj[w, t, :] = pos

    return traj

def turn_coord_to_index(
        coord: float,
        lo: float,
        hi: float,
        n_bins: int,
) -> int:
    """
    Map a coordinate to a grid index in [0, n_bins-1].
    Handles possible floating-point edge cases by clamping.
    """
    if hi <= lo:
        raise ValueError("hi must be > lo")
    # normalise to [0, 1)
    u = (coord - lo) / (hi - lo)
    # map to [0, n_bins)
    i = int(u * n_bins)
    # clamp to valid range
    if i < 0:
        i = 0
    elif i >= n_bins:
        i = n_bins - 1
    return i

def turn_position_to_grid_index(
        box: "SimulationBox3D",
        pos: np.ndarray,
        n_grid: int,
) -> tuple[int, int, int]:
    """
    Turns a 3D position into grid index
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    ix = turn_coord_to_index(x, box.xmin, box.xmax, n_grid)
    iy = turn_coord_to_index(y, box.ymin, box.ymax, n_grid)
    iz = turn_coord_to_index(z, box.zmin, box.zmax, n_grid)
    return ix, iy, iz

def estimate_accessible_volume_random_walk(
        box: "SimulationBox3D",
        spheres: Sequence["Sphere"],
        num_walkers: int,
        num_steps: int,
        step_sigma: float,
        n_grid: int = 32,
        rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """
    Estimates accessible volume of box by having many random walkers and a 3D occupancy grid
    """
    # Input validation
    if n_grid <= 0:
        raise ValueError("n_grid must be > 0")

    rng = rng or np.random.default_rng()

    # Simulate random walkers and get their trajectories
    traj = random_walkers_avoiding_spheres(
        box=box,
        spheres=spheres,
        num_walkers=num_walkers,
        num_steps=num_steps,
        step_sigma=step_sigma,
        rng=rng,
    )

    # Initialize 3D occupancy grid. Visited tells if a grid cell has been traveled through by a walker
    visited = np.zeros((n_grid, n_grid, n_grid), dtype=bool)

    # loop over all walkers and time steps, and update visited trajectory paths to true
    for w in range(num_walkers):
        for t in range(num_steps + 1):
            ix, iy, iz = turn_position_to_grid_index(box, traj[w, t, :], n_grid)
            visited[ix, iy, iz] = True

    # Count visited cells vs total cells and find the accessible fraction of the box
    num_visited = int(visited.sum())
    num_total = visited.size
    accessible_fraction = num_visited / float(num_total)
    stderr_fraction = math.sqrt(
        max(accessible_fraction * (1.0 - accessible_fraction), 0.0) / num_total
    )

    # Converting fraction to volume
    volume_box = box.volume()
    accessible_volume = accessible_fraction * volume_box
    stderr_volume = stderr_fraction * volume_box

    return accessible_fraction, stderr_fraction, accessible_volume, stderr_volume

def compare_dna_accessible_box_volumes(
        box: "SimulationBox3D",
        dna_volume: float,
        accessible_volume: float,
) -> None:
    """
    description:
        Print a comparison between DNA volume, accessible volume,
        and the total simulation box volume.
    args:
        box (SimulationBox3D): 3D simulation box used for DNA and random walks.
        dna_volume (float): Estimated DNA volume (from Monte Carlo in Topic 1).
        accessible_volume (float): Estimated accessible volume (from random walks in Topic 2).
    returns:
        Prints numbers 
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Basic volumes
    V_box = box.volume()
    V_DNA = float(dna_volume)
    V_acc = float(accessible_volume)
    V_gap = V_box - (V_DNA + V_acc)

    # Prints
    print(f"Box volume        : {V_box:.1f} Å³")
    print(f"DNA volume (est.) : {V_DNA:.1f} Å³")
    print(f"Accessible volume : {V_acc:.1f} Å³")
    print(f"Residual gap      : {V_gap:.1f} Å³  "
          f"({100.0 * V_gap / V_box:.1f}% of box)\n")


def known_volume_test_accessible_volume_from_walkers() -> dict[str, float]:
    """
    description:
        Toy sanity test for the random-walk accessible volume method on a simple
        geometry where we know the analytic solution: one sphere in a cube.
        Uses estimate_accessible_volume_from_walkers and checks that the
        estimated accessible volume is reasonably close to V_box - V_sphere.
    args:
        None.
    returns:
        dict with keys:
            "V_box"          : box volume (Å^3)
            "V_sphere"       : sphere volume (Å^3)
            "V_acc_analytic" : analytic accessible volume = V_box - V_sphere (Å^3)
            "frac_acc_rw"    : random-walk accessible fraction
            "stderr_frac_rw" : standard error of accessible fraction
            "V_acc_rw"       : random-walk accessible volume (Å^3)
            "stderr_V_rw"    : standard error on volume (Å^3)
            "rel_error"      : relative error |V_rw - V_analytic| / V_analytic
    """
    # Simple cube box
    box = SimulationBox3D(
        x=(0.0, 10.0),
        y=(0.0, 10.0),
        z=(0.0, 10.0),
    )

    # One sphere in the center
    r = 3.0
    sphere = Sphere(center=(5.0, 5.0, 5.0), radius=r)
    spheres = [sphere]

    # Analytic reference volumes
    V_box = box.volume()
    V_sphere = sphere.volume()
    V_acc_analytic = V_box - V_sphere

    # Random-walk estimate of accessible volume
    rng = np.random.default_rng(123)
    frac_acc_rw, stderr_frac_rw, V_acc_rw, stderr_V_rw = (
        estimate_accessible_volume_random_walk(
            box=box,
            spheres=spheres,
            num_walkers=500,
            num_steps=5000,
            step_sigma=0.5,
            n_grid=32,
            rng=rng,
        )
    )

    # Relative error vs analytic solution
    rel_error = abs(V_acc_rw - V_acc_analytic) / V_acc_analytic

    # Print a short summary
    print(f"Known volume test: cube 10×10×10 Å with one sphere (r=3 Å) in center")
    print(f"Box volume         = {V_box:.3f} Å³")
    print(f"Sphere volume      = {V_sphere:.3f} Å³")
    print(f"Analytic V_access  = {V_acc_analytic:.3f} Å³\n")

    print(f"RW accessible frac ≈ {frac_acc_rw:.4f} ± {2*stderr_frac_rw:.4f} (≈95% CI)")
    print(f"RW accessible vol  ≈ {V_acc_rw:.2f} ± {2*stderr_V_rw:.2f} Å³")
    print(f"Relative error     = {rel_error:.3%}")

    # Loose assertion: should be in the right ballpark
    assert rel_error < 0.3, (
        "Random-walk accessible volume is too far from analytic value in toy test."
    )
    print("Toy test passed ✅ (RW estimate consistent with analytic geometry)")

    return {
        "V_box": V_box,
        "V_sphere": V_sphere,
        "V_acc_analytic": V_acc_analytic,
        "frac_acc_rw": frac_acc_rw,
        "stderr_frac_rw": stderr_frac_rw,
        "V_acc_rw": V_acc_rw,
        "stderr_V_rw": stderr_V_rw,
        "rel_error": rel_error,
    }