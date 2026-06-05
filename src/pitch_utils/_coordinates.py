from dataclasses import dataclass
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
    def __init__(self, min_point: Point, max_point: Point) -> None:
        self._min_point = min_point
        self._max_point = max_point
        self._point_1 = min_point
        self._point_4 = max_point

    @property
    def _point_2(self) -> Point:
        return Point(x=self._max_point.x, y=self._min_point.y)

    @property
    def _point_3(self) -> Point:
        return Point(x=self._min_point.x, y=self._max_point.y)

    @property
    def min_point(self) -> Point:
        return self._min_point

    @property
    def max_point(self) -> Point:
        return self._max_point

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Rectangle):
            return False
        return self.coords == value.coords

    def __repr__(self) -> str:
        return f"Rectangle(coords={self.coords})"

    def center(self) -> Point:
        return Point(
            x=(self._min_point.x + self._max_point.x) / 2,
            y=(self._min_point.y + self._max_point.y) / 2,
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
        return (
            self._point_1.coords,
            self._point_2.coords,
            self._point_3.coords,
            self._point_4.coords,
        )


@dataclass(frozen=True)
class CoordinateSystem:
    origin: tuple[float, float]
    x_dir: Literal["left", "right"]
    y_dir: Literal["up", "down"]

    def shift_x(self, x: float, offset: float) -> float:
        return x + offset if self.x_dir == "right" else x - offset

    def shift_y(self, y: float, offset: float) -> float:
        return y + offset if self.y_dir == "up" else y - offset

    @overload
    def reflect(self, geom: Point, pivot: Line) -> Point: ...

    @overload
    def reflect(self, geom: Line, pivot: Line) -> Line: ...

    @overload
    def reflect(self, geom: Circle, pivot: Line) -> Circle: ...

    @overload
    def reflect(self, geom: Rectangle, pivot: Line) -> Rectangle: ...

    def reflect(self, geom, pivot):
        raise NotImplementedError

    def rotate(self, geom, angle: float, origin: "Point") -> "Point":
        raise NotImplementedError
