
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
    points.append(Point(0.0 * sx, 0.5625 * sy, False)) # P2
    points.append(Point(0.84 * sx, 0.5625 * sy, True))  # P3
    points.append(Point(0.0 * sx, 0.375 * sy, False)) # P4
    points.append(Point(0.227 * sx, 0.375 * sy, True)) # P5
    points.append(Point(0.227 * sx, 0.0 * sy, False)) # P6
    points.append(Point(0.227 * sx, 0.5625 * sy, True)) # P7
    points.append(Point(0.522 * sx, 0.0 * sy, False)) # P8
    points.append(Point(0.522 * sx, 0.5625 * sy, True)) # P9
    points.append(Point(0.84 * sx, 0.0 * sy, False)) # P10
    points.append(Point(0.84 * sx, 0.6875 * sy, True)) # P11

    return points