from __future__ import annotations

import math
from functools import total_ordering
from typing import Any, Iterator, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt


def _scale(
    values: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if np.any(factors <= 0):
        raise ValueError("Scale factors must be positive")
    return origin + (values - origin) * factors


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

    def scaled(
        self, x_factor: float, y_factor: float, origin: Point
    ) -> Points:
        return Points.from_numpy(
            _scale(
                self.to_numpy(),
                np.array([x_factor, y_factor], dtype=np.float64),
                np.array(origin.coords, dtype=np.float64),
            )
        )


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

    def to_points(self) -> Points:
        return Points.from_points([self])

    def scaled(self, x_factor: float, y_factor: float, origin: Point) -> Point:
        coordinates = _scale(
            np.array([self.coords], dtype=np.float64),
            np.array([x_factor, y_factor], dtype=np.float64),
            np.array(origin.coords, dtype=np.float64),
        )
        return Point(*coordinates[0])

    @classmethod
    def from_points(cls, points: Points) -> Point:
        return points.to_points()[0]


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

    def to_points(self) -> Points:
        return Points.from_points([self.start, self.end])

    def scaled(self, x_factor: float, y_factor: float, origin: Point) -> Line:
        return Line.from_points(
            Points.from_numpy(
                _scale(
                    self.to_points().to_numpy(),
                    np.array([x_factor, y_factor], dtype=np.float64),
                    np.array(origin.coords, dtype=np.float64),
                )
            )
        )

    @classmethod
    def from_points(cls, points: Points) -> Line:
        transformed = points.to_points()
        return cls(transformed[0], transformed[1])


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

    def to_points(self) -> Points:
        return Points.from_points([self.center])

    def with_points(self, points: Points) -> Circle:
        return Circle(Point.from_points(points), self.radius)

    def scaled(self, factor: float, origin: Point) -> Circle:
        center = _scale(
            self.to_points().to_numpy(),
            np.array([factor, factor], dtype=np.float64),
            np.array(origin.coords, dtype=np.float64),
        )
        radius = _scale(
            np.array([self.radius], dtype=np.float64),
            np.array([factor], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
        return Circle(Point(*center[0]), float(radius[0]))


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

    def to_points(self) -> Points:
        return Points.from_rows(self.coords)

    def scaled(
        self, x_factor: float, y_factor: float, origin: Point
    ) -> Rectangle:
        return Rectangle.from_points(
            Points.from_numpy(
                _scale(
                    self.to_points().to_numpy(),
                    np.array([x_factor, y_factor], dtype=np.float64),
                    np.array(origin.coords, dtype=np.float64),
                )
            )
        )

    @classmethod
    def from_points(cls, points: Points) -> Rectangle:
        coordinates = points.to_numpy()
        return cls(
            Point(
                float(coordinates[:, 0].min()), float(coordinates[:, 1].min())
            ),
            Point(
                float(coordinates[:, 0].max()), float(coordinates[:, 1].max())
            ),
        )


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

    def flipped(self) -> DirectedRange:
        return DirectedRange(self.end, self.start)

    def scaled(self, factor: float, origin: float) -> DirectedRange:
        values = _scale(
            np.array([[self.start], [self.end]], dtype=np.float64),
            np.array([factor], dtype=np.float64),
            np.array([origin], dtype=np.float64),
        )
        return DirectedRange(float(values[0, 0]), float(values[1, 0]))


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

    @staticmethod
    def _offset_sign(op: Literal["+", "-"]) -> Literal[-1, 1]:
        if op == "+":
            return 1
        if op == "-":
            return -1
        raise ValueError("Operation must be '+' or '-'")

    def _shift_points(
        self,
        geom: Points,
        x_offset: float,
        y_offset: float,
        x_op: Literal["+", "-"],
        y_op: Literal["+", "-"],
    ) -> Points:
        x_shift = x_offset * self._x_sign * self._offset_sign(x_op)
        y_shift = y_offset * self._y_sign * self._offset_sign(y_op)
        return Points.from_numpy(
            geom.to_numpy() + np.array([x_shift, y_shift])
        )

    def _reflect_points(self, geom: Points, pivot: Line) -> Points:
        pivot_start = np.array(pivot.start.coords)
        pivot_vector = np.array(
            [pivot.end.x - pivot.start.x, pivot.end.y - pivot.start.y]
        )
        coordinates = geom.to_numpy()
        projection_scale = (
            (coordinates - pivot_start) @ pivot_vector
        ) / np.dot(pivot_vector, pivot_vector)
        projection = pivot_start + (
            projection_scale[:, np.newaxis] * pivot_vector
        )
        return Points.from_numpy(np.round((2 * projection) - coordinates, 2))

    def _rotate_points(
        self, geom: Points, angle: float, origin: Point
    ) -> Points:
        origin_coords = np.array(origin.coords)
        signs = np.array([self._x_sign, self._y_sign])
        standard = (geom.to_numpy() - origin_coords) * signs
        radians = np.deg2rad(angle)
        rotation = np.array(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ]
        )
        rotated = (standard @ rotation.T * signs) + origin_coords
        return Points.from_numpy(np.round(rotated, 2))

    def scaled(
        self, x_factor: float, y_factor: float, origin: Point
    ) -> CoordinateSystem:
        return CoordinateSystem(
            origin=self.origin.scaled(x_factor, y_factor, origin),
            x_range=self.x_range.scaled(x_factor, origin.x),
            y_range=self.y_range.scaled(y_factor, origin.y),
        )

    def flipped(
        self, *, x: bool = False, y: bool = False
    ) -> CoordinateSystem:
        return CoordinateSystem(
            origin=Point(
                (
                    self.x_range.start
                    + self.x_range.end
                    - self.origin.x
                    if x
                    else self.origin.x
                ),
                (
                    self.y_range.start
                    + self.y_range.end
                    - self.origin.y
                    if y
                    else self.origin.y
                ),
            ),
            x_range=self.x_range.flipped() if x else self.x_range,
            y_range=self.y_range.flipped() if y else self.y_range,
        )

    @overload
    def shift(
        self,
        geom: Points,
        x_offset: float = 0,
        y_offset: float = 0,
        x_op: Literal["+", "-"] = "+",
        y_op: Literal["+", "-"] = "+",
    ) -> Points: ...

    @overload
    def shift(
        self,
        geom: Point,
        x_offset: float = 0,
        y_offset: float = 0,
        x_op: Literal["+", "-"] = "+",
        y_op: Literal["+", "-"] = "+",
    ) -> Point: ...

    @overload
    def shift(
        self,
        geom: Line,
        x_offset: float = 0,
        y_offset: float = 0,
        x_op: Literal["+", "-"] = "+",
        y_op: Literal["+", "-"] = "+",
    ) -> Line: ...

    @overload
    def shift(
        self,
        geom: Circle,
        x_offset: float = 0,
        y_offset: float = 0,
        x_op: Literal["+", "-"] = "+",
        y_op: Literal["+", "-"] = "+",
    ) -> Circle: ...

    @overload
    def shift(
        self,
        geom: Rectangle,
        x_offset: float = 0,
        y_offset: float = 0,
        x_op: Literal["+", "-"] = "+",
        y_op: Literal["+", "-"] = "+",
    ) -> Rectangle: ...

    def shift(self, geom, x_offset=0, y_offset=0, x_op="+", y_op="+"):
        points = geom if isinstance(geom, Points) else geom.to_points()
        transformed = self._shift_points(
            points, x_offset, y_offset, x_op, y_op
        )
        if isinstance(geom, Points):
            return transformed
        if isinstance(geom, Circle):
            return geom.with_points(transformed)
        return type(geom).from_points(transformed)

    @overload
    def reflect(self, geom: Point, pivot: Line) -> Point: ...

    @overload
    def reflect(self, geom: Points, pivot: Line) -> Points: ...

    @overload
    def reflect(self, geom: Line, pivot: Line) -> Line: ...

    @overload
    def reflect(self, geom: Circle, pivot: Line) -> Circle: ...

    @overload
    def reflect(self, geom: Rectangle, pivot: Line) -> Rectangle: ...

    def reflect(self, geom, pivot):
        points = geom if isinstance(geom, Points) else geom.to_points()
        transformed = self._reflect_points(points, pivot)
        if isinstance(geom, Points):
            return transformed
        if isinstance(geom, Circle):
            return geom.with_points(transformed)
        return type(geom).from_points(transformed)

    @overload
    def rotate(self, geom: Point, angle: float, origin: Point) -> Point: ...

    @overload
    def rotate(self, geom: Points, angle: float, origin: Point) -> Points: ...

    @overload
    def rotate(self, geom: Line, angle: float, origin: Point) -> Line: ...

    @overload
    def rotate(self, geom: Circle, angle: float, origin: Point) -> Circle: ...

    @overload
    def rotate(
        self, geom: Rectangle, angle: float, origin: Point
    ) -> Rectangle: ...

    def rotate(self, geom, angle, origin):
        points = geom if isinstance(geom, Points) else geom.to_points()
        transformed = self._rotate_points(points, angle, origin)
        if isinstance(geom, Points):
            return transformed
        if isinstance(geom, Circle):
            return geom.with_points(transformed)
        return type(geom).from_points(transformed)
