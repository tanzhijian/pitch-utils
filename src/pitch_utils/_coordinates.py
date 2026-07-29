from __future__ import annotations

import math
from functools import total_ordering
from typing import Any, Iterator, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt


class Points:
    def __init__(self, arr: npt.NDArray[np.float64]) -> None:
        self._validate_shape(arr)
        self._arr = arr

    @staticmethod
    def _validate_shape(arr: npt.NDArray[np.float64]) -> None:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Invalid shape {arr.shape}. Data must be a 2D "
                "array-like structure with shape (N, 2)."
            )

    @classmethod
    def from_points(cls, points: Sequence[Point]) -> Points:
        arr = np.array(
            [(point.x, point.y) for point in points], dtype=np.float64
        )
        return cls(arr)

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[float]]) -> Points:
        arr = np.array(rows, dtype=np.float64)
        return cls(arr)

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.number]) -> Points:
        arr = np.asarray(array, dtype=np.float64)
        return cls(arr)

    @classmethod
    def from_columns(
        cls,
        data: dict[str, Any] | Any,
        x_col: str,
        y_col: str,
    ) -> Points:
        raise NotImplementedError

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self._arr.copy()

    def to_points(self) -> list[Point]:
        return [Point(x, y) for x, y in self._arr]

    def to_list(self) -> list[list[float]]:
        return self._arr.tolist()

    def iter_tuples(self) -> Iterator[tuple[float, float]]:
        for row in self._arr:
            yield (row[0], row[1])


@total_ordering
class Point:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.coords < other.coords

    def __repr__(self) -> str:
        return f"Point(x={self._x}, y={self._y})"

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def coords(self) -> tuple[float, float]:
        return (self._x, self._y)


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        if start == end:
            raise ValueError("Start and end points cannot be the same")
        self._start = start
        self._end = end

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Line):
            return False
        normal_match = self.start == value.start and self.end == value.end
        reverse_match = self.start == value.end and self.end == value.start
        return normal_match or reverse_match

    def __repr__(self) -> str:
        return f"Line(start={self._start}, end={self._end})"

    def is_strictly_equal(self, other: Line) -> bool:
        return self.coords == other.coords

    @property
    def start(self) -> Point:
        return self._start

    @property
    def end(self) -> Point:
        return self._end

    @property
    def length(self) -> float:
        return (
            (self._end.x - self._start.x) ** 2
            + (self._end.y - self._start.y) ** 2
        ) ** 0.5

    @property
    def center(self) -> Point:
        return Point(
            x=(self._start.x + self._end.x) / 2,
            y=(self._start.y + self._end.y) / 2,
        )

    @property
    def coords(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self._start.coords, self._end.coords)


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self._center = center
        self._radius = radius

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Circle):
            return False
        return self.center == value.center and self.radius == value.radius

    def __repr__(self) -> str:
        return f"Circle(center={self._center}, radius={self._radius})"

    @property
    def center(self) -> Point:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius


class Rectangle:
    def __init__(self, p1: Point, p2: Point) -> None:
        if p1.x == p2.x or p1.y == p2.y:
            raise ValueError(
                "Points cannot be on the same vertical or horizontal line"
            )
        self._p1 = p1
        self._p2 = p2

    @property
    def _p3(self) -> Point:
        return Point(x=self._p1.x, y=self._p2.y)

    @property
    def _p4(self) -> Point:
        return Point(x=self._p2.x, y=self._p1.y)

    @property
    def min_point(self) -> Point:
        return Point(
            x=min(self._p1.x, self._p2.x),
            y=min(self._p1.y, self._p2.y),
        )

    @property
    def max_point(self) -> Point:
        return Point(
            x=max(self._p1.x, self._p2.x),
            y=max(self._p1.y, self._p2.y),
        )

    @property
    def points(self) -> tuple[Point, Point, Point, Point]:
        s1, s2, s3, s4 = sorted((self._p1, self._p2, self._p3, self._p4))
        return (s1, s2, s3, s4)

    @property
    def coords(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        s1, s2, s3, s4 = self.points
        return (s1.coords, s2.coords, s3.coords, s4.coords)

    @property
    def center(self) -> Point:
        return Point(
            x=(self._p1.x + self._p2.x) / 2,
            y=(self._p1.y + self._p2.y) / 2,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Rectangle):
            return False
        return self.points == value.points

    def __repr__(self) -> str:
        return f"Rectangle(coords={self.coords})"


class DirectedRange:
    def __init__(self, start: float, end: float) -> None:
        self._start = start
        self._end = end

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DirectedRange):
            return False
        return math.isclose(self.start, value.start) and math.isclose(
            self.end, value.end
        )

    def __repr__(self) -> str:
        return f"DirectedRange(start={self._start}, end={self._end})"

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def coords(self) -> tuple[float, float]:
        return (self._start, self._end)

    @property
    def length(self) -> float:
        return abs(self._end - self._start)


class CoordinateSystem:
    def __init__(
        self,
        origin: Point,
        x_range: DirectedRange,
        y_range: DirectedRange,
    ) -> None:
        self._origin = origin
        self._x_range = x_range
        self._y_range = y_range

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CoordinateSystem):
            return False
        return (
            self.origin == value.origin
            and self.x_range == value.x_range
            and self.y_range == value.y_range
        )

    def __repr__(self) -> str:
        return (
            f"CoordinateSystem(origin={self._origin}, "
            f"x_range={self._x_range}, y_range={self._y_range}, "
            f"x_dir='{self.x_dir}', y_dir='{self.y_dir}')"
        )

    @property
    def origin(self) -> Point:
        return self._origin

    @property
    def x_range(self) -> DirectedRange:
        return self._x_range

    @property
    def y_range(self) -> DirectedRange:
        return self._y_range

    @property
    def x_dir(self) -> Literal["left", "right"]:
        return "right" if self._x_range.end > self._x_range.start else "left"

    @property
    def y_dir(self) -> Literal["up", "down"]:
        return "up" if self._y_range.end > self._y_range.start else "down"

    @property
    def _x_sign(self) -> Literal[-1, 1]:
        return 1 if self.x_dir == "right" else -1

    @property
    def _y_sign(self) -> Literal[-1, 1]:
        return 1 if self.y_dir == "up" else -1

    def shift_x(
        self,
        x: float,
        *offsets: float,
        op: Literal["+", "-"] = "+",
    ) -> float:
        op_sign = 1 if op == "+" else -1
        return x + (sum(offsets) * self._x_sign * op_sign)

    def shift_y(
        self,
        y: float,
        *offsets: float,
        op: Literal["+", "-"] = "+",
    ) -> float:
        op_sign = 1 if op == "+" else -1
        return y + (sum(offsets) * self._y_sign * op_sign)

    def _normalize_value(self, value: float) -> float:
        return round(value, 2)

    def _reflect_point(self, geom: Point, pivot: Line) -> Point:
        pivot_dx = pivot.end.x - pivot.start.x
        pivot_dy = pivot.end.y - pivot.start.y
        pivot_length_sq = (pivot_dx**2) + (pivot_dy**2)

        start_to_point_x = geom.x - pivot.start.x
        start_to_point_y = geom.y - pivot.start.y
        projection_scale = (
            (start_to_point_x * pivot_dx) + (start_to_point_y * pivot_dy)
        ) / pivot_length_sq
        projection_x = pivot.start.x + (projection_scale * pivot_dx)
        projection_y = pivot.start.y + (projection_scale * pivot_dy)

        reflected_x = (2 * projection_x) - geom.x
        reflected_y = (2 * projection_y) - geom.y
        return Point(
            x=self._normalize_value(reflected_x),
            y=self._normalize_value(reflected_y),
        )

    def _reflect_line(self, geom: Line, pivot: Line) -> Line:
        return Line(
            start=self._reflect_point(geom.start, pivot),
            end=self._reflect_point(geom.end, pivot),
        )

    def _reflect_circle(self, geom: Circle, pivot: Line) -> Circle:
        return Circle(
            center=self._reflect_point(geom.center, pivot),
            radius=geom.radius,
        )

    def _reflect_rectangle(self, geom: Rectangle, pivot: Line) -> Rectangle:
        reflected_coords = [
            self._reflect_point(Point(x, y), pivot).coords
            for x, y in geom.coords
        ]
        xs = [x for x, _ in reflected_coords]
        ys = [y for _, y in reflected_coords]
        return Rectangle(
            p1=Point(min(xs), min(ys)),
            p2=Point(max(xs), max(ys)),
        )

    @overload
    def reflect(self, geom: Point, pivot: Line) -> Point: ...

    @overload
    def reflect(self, geom: Line, pivot: Line) -> Line: ...

    @overload
    def reflect(self, geom: Circle, pivot: Line) -> Circle: ...

    @overload
    def reflect(self, geom: Rectangle, pivot: Line) -> Rectangle: ...

    def reflect(self, geom, pivot):
        if isinstance(geom, Point):
            return self._reflect_point(geom, pivot)
        elif isinstance(geom, Line):
            return self._reflect_line(geom, pivot)
        elif isinstance(geom, Circle):
            return self._reflect_circle(geom, pivot)
        elif isinstance(geom, Rectangle):
            return self._reflect_rectangle(geom, pivot)
        else:
            raise TypeError("Unsupported geometry type")

    def _rotate_point(self, geom: Point, angle: float, origin: Point) -> Point:
        translated_x = geom.x - origin.x
        translated_y = geom.y - origin.y

        # Convert to a standard right/up plane so positive angles stay
        # counterclockwise in this coordinate system's orientation.
        standard_x = translated_x * self._x_sign
        standard_y = translated_y * self._y_sign
        radians = np.deg2rad(angle)

        cos_angle = np.cos(radians)
        sin_angle = np.sin(radians)
        rotated_x = (standard_x * cos_angle) - (standard_y * sin_angle)
        rotated_y = (standard_x * sin_angle) + (standard_y * cos_angle)

        x = origin.x + (rotated_x * self._x_sign)
        y = origin.y + (rotated_y * self._y_sign)

        return Point(
            x=self._normalize_value(x),
            y=self._normalize_value(y),
        )

    def _rotate_line(self, geom: Line, angle: float, origin: Point) -> Line:
        return Line(
            start=self._rotate_point(geom.start, angle, origin),
            end=self._rotate_point(geom.end, angle, origin),
        )

    def _rotate_circle(
        self, geom: Circle, angle: float, origin: Point
    ) -> Circle:
        return Circle(
            center=self._rotate_point(geom.center, angle, origin),
            radius=geom.radius,
        )

    def _rotate_rectangle(
        self, geom: Rectangle, angle: float, origin: Point
    ) -> Rectangle:
        rotated_coords = [
            self._rotate_point(Point(x, y), angle, origin).coords
            for x, y in geom.coords
        ]
        xs = [x for x, _ in rotated_coords]
        ys = [y for _, y in rotated_coords]
        return Rectangle(
            p1=Point(min(xs), min(ys)),
            p2=Point(max(xs), max(ys)),
        )

    @overload
    def rotate(self, geom: Point, angle: float, origin: Point) -> Point: ...

    @overload
    def rotate(self, geom: Line, angle: float, origin: Point) -> Line: ...

    @overload
    def rotate(self, geom: Circle, angle: float, origin: Point) -> Circle: ...

    @overload
    def rotate(
        self, geom: Rectangle, angle: float, origin: Point
    ) -> Rectangle: ...

    def rotate(self, geom, angle, origin):
        if isinstance(geom, Point):
            return self._rotate_point(geom, angle, origin)
        elif isinstance(geom, Line):
            return self._rotate_line(geom, angle, origin)
        elif isinstance(geom, Circle):
            return self._rotate_circle(geom, angle, origin)
        elif isinstance(geom, Rectangle):
            return self._rotate_rectangle(geom, angle, origin)
        else:
            raise TypeError("Unsupported geometry type")
