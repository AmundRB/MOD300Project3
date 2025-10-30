class SimulationBox3D:
    """Axis-aligned 3D simulation box (Å)."""

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("Invalid bounds: require xmin<xmax, ymin<ymax, zmin<zmax")

    def volume(self):
        """Calculate the volume of the simulation box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    def surface_area(self):
        """Calculate the surface area of the simulation box."""
        return 2 * ((self.xmax - self.xmin) * (self.ymax - self.ymin) +
                     (self.ymax - self.ymin) * (self.zmax - self.zmin) +
                     (self.zmax - self.zmin) * (self.xmax - self.xmin))

    def is_point_inside(self, x, y, z):
        """Check if a point (x, y, z) is inside the simulation box."""
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)
