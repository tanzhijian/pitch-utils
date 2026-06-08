from functools import total_ordering
from typing import Iterator, Literal, overload

import numpy as np
import numpy.typing as npt

from ._types import LocationsTypes


class Locations:
    def __init__(
        self,
        data: LocationsTypes,
    ) -> None:
        self._arr = self._to_array(data)

    def _to_array(
        self,
        data: LocationsTypes,
    ) -> npt.NDArray[np.float64]:
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError
            arr = arr[np.newaxis, :]

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError

        return arr

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self._arr

    def to_list(self) -> list[list[float]]:
        return self._arr.tolist()

    def iter_tuples(self) -> Iterator[tuple[float, float]]:
        for row in self._arr:
            yield tuple(row)


@total_ordering
class Point:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Point):
            return False
        return self.coords == value.coords

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
        self._start = start
        self._end = end

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Line):
            return False
        normal_match = self._start == value.start and self._end == value.end
        reverse_match = self._start == value.end and self._end == value.start
        return normal_match or reverse_match

    def __repr__(self) -> str:
        return f"Line(start={self._start}, end={self._end})"

    def is_strictly_equal(self, other: "Line") -> bool:
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
        return (self.start.coords, self.end.coords)


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self._center = center
        self._radius = radius

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Circle):
            return False
        return self._center == value._center and self._radius == value._radius

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
    def coords(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        s1, s2, s3, s4 = sorted(
            (
                self._p1.coords,
                self._p2.coords,
                self._p3.coords,
                self._p4.coords,
            )
        )
        return (s1, s2, s3, s4)

    def center(self) -> Point:
        return Point(
            x=(self._p1.x + self._p2.x) / 2,
            y=(self._p1.y + self._p2.y) / 2,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Rectangle):
            return False
        return self.coords == value.coords

    def __repr__(self) -> str:
        return f"Rectangle(coords={self.coords})"


class CoordinateSystem:
    def __init__(
        self,
        origin: tuple[float, float],
        x_dir: Literal["left", "right"],
        y_dir: Literal["up", "down"],
    ) -> None:
        self._origin = origin
        self._x_dir = x_dir
        self._y_dir = y_dir

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CoordinateSystem):
            return False
        return (
            self._origin == value._origin
            and self._x_dir == value._x_dir
            and self._y_dir == value._y_dir
        )

    def __repr__(self) -> str:
        return (
            f"CoordinateSystem(origin={self._origin}, "
            f"x_dir='{self._x_dir}', y_dir='{self._y_dir}')"
        )

    @property
    def origin(self) -> tuple[float, float]:
        return self._origin

    @property
    def x_dir(self) -> Literal["left", "right"]:
        return self._x_dir

    @property
    def y_dir(self) -> Literal["up", "down"]:
        return self._y_dir

    @property
    def _x_sign(self) -> Literal[-1, 1]:
        return 1 if self._x_dir == "right" else -1

    @property
    def _y_sign(self) -> Literal[-1, 1]:
        return 1 if self._y_dir == "up" else -1

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
        rounded = round(value, 2)
        if np.isclose(value, rounded):
            value = rounded
        return float(value)

    def _reflect_point(self, geom: Point, pivot: Line) -> Point:
        pivot_dx = pivot.end.x - pivot.start.x
        pivot_dy = pivot.end.y - pivot.start.y
        pivot_length_sq = (pivot_dx**2) + (pivot_dy**2)

        if pivot_length_sq == 0:
            raise ValueError("Pivot line cannot have zero length")

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
