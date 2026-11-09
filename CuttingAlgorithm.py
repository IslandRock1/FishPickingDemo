
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    cut: bool

def getCutPath(size: tuple[int, int]) -> list[Point]:
    points = []
    sx, sy = size

    # Starting at bottom left corner
    points.append(Point(0.2 * sx, 0.2 * sy, False)) # Moving to first cut point
    points.append(Point(0.8 * sx, 0.2 * sy, True))  # Executing first cut
    # Etc, etc. My points are obviously wrong.

    return points