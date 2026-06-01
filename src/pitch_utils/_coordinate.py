from typing import Literal, overload


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def rotate90_around(self, center: "Point") -> "Point":
        raise NotImplementedError


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end

    @property
    def length(self) -> float:
        return (
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2
        ) ** 0.5


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius


class Rectangle:
    def __init__(
        self,
        bottom_left: Point,
        width: float,
        height: float,
    ) -> None:
        self.bottom_left = bottom_left
        self.width = width
        self.height = height


class CoordinateSystem:
    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        origin: Point = Point(0, 0),
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.origin = origin

    def _validate_axis(self, axis: tuple[float, float]) -> None:
        if len(axis) != 2:
            raise ValueError("Range must be a tuple of length 2")
        start, end = axis
        if not isinstance(start, (int, float)) or not isinstance(
            end, (int, float)
        ):
            raise TypeError("Range values must be numeric")
        if start == end:
            raise ValueError("Range values cannot be the same")

    @property
    def x_dir(self) -> Literal["left", "right"]:
        x0, x1 = self.x_range
        if x1 > x0:
            return "right"
        else:
            return "left"

    @property
    def y_dir(self) -> Literal["up", "down"]:
        y0, y1 = self.y_range
        if y1 > y0:
            return "up"
        else:
            return "down"

    @property
    def center(self) -> Point:
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        return Point(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
        )

    @property
    def x_length(self) -> float:
        x0, x1 = self.x_range
        return abs(x1 - x0)

    @property
    def y_length(self) -> float:
        y0, y1 = self.y_range
        return abs(y1 - y0)

    @overload
    def reflect(self, geometry: Point, line: Line) -> Point: ...

    @overload
    def reflect(self, geometry: Line, line: Line) -> Line: ...

    @overload
    def reflect(self, geometry: Circle, line: Line) -> Circle: ...

    @overload
    def reflect(self, geometry: Rectangle, line: Line) -> Rectangle: ...

    def reflect(self, geometry, line):
        raise NotImplementedError
