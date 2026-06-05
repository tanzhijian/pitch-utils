from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from typing import Literal

from ._markings import Markings


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

    def rotate(self, angle: float, origin: "Point") -> "Point":
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Point":
        raise NotImplementedError


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

    @property
    def is_vertical(self) -> bool:
        raise NotImplementedError

    def rotate(self, angle: float, origin: Point) -> Point:
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Line":
        raise NotImplementedError


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

    def rotate(self, angle: float, origin: Point) -> Point:
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Circle":
        raise NotImplementedError


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

    def rotate(self, angle: float, origin: Point) -> Point:
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Rectangle":
        raise NotImplementedError


@dataclass(frozen=True)
class CoordinateSystem:
    origin: tuple[float, float]
    x_dir: Literal["left", "right"]
    y_dir: Literal["up", "down"]

    def shift_x(self, x: float, offset: float) -> float:
        return x + offset if self.x_dir == "right" else x - offset

    def shift_y(self, y: float, offset: float) -> float:
        return y + offset if self.y_dir == "up" else y - offset


class Pitch(ABC):
    def __init__(
        self,
        touch_line_range: tuple[float, float],
        goal_line_range: tuple[float, float],
        markings: Markings | None = None,
        coord_sys: CoordinateSystem | None = None,
    ) -> None:
        self._touch_line_range = touch_line_range
        self._goal_line_range = goal_line_range

        touch_line = abs(touch_line_range[1] - touch_line_range[0])
        goal_line = abs(goal_line_range[1] - goal_line_range[0])
        self._markings = (
            markings
            if markings is not None
            else Markings(touch_line=touch_line, goal_line=goal_line)
        )

        self._coord_sys = (
            coord_sys if coord_sys is not None else self._build_coord_sys()
        )

    @abstractmethod
    def _build_coord_sys(self) -> CoordinateSystem:
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if type(value) is not type(self):
            return False
        return self.coords == value.coords and self.markings == value.markings

    @property
    def coord_sys(self) -> CoordinateSystem:
        return self._coord_sys

    @property
    def markings(self) -> Markings:
        return self._markings

    @property
    @abstractmethod
    def bottom_left(self) -> Point:
        raise NotImplementedError

    @property
    @abstractmethod
    def bottom_right(self) -> Point:
        raise NotImplementedError

    @property
    @abstractmethod
    def top_left(self) -> Point:
        raise NotImplementedError

    @property
    @abstractmethod
    def top_right(self) -> Point:
        raise NotImplementedError

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
            self.bottom_left.coords,
            self.bottom_right.coords,
            self.top_left.coords,
            self.top_right.coords,
        )

    @property
    def left(self) -> Line:
        return Line(self.bottom_left, self.top_left)

    @property
    def right(self) -> Line:
        return Line(self.bottom_right, self.top_right)

    @property
    def bottom(self) -> Line:
        return Line(self.bottom_left, self.bottom_right)

    @property
    def top(self) -> Line:
        return Line(self.top_left, self.top_right)

    @property
    def centre_circle(self) -> Circle:
        return Circle(
            center=self.halfway_line.center,
            radius=self._markings.centre_circle_radius,
        )

    @property
    @abstractmethod
    def halfway_line(self) -> Line:
        raise NotImplementedError

    @property
    def centre_mark(self) -> Circle:
        return Circle(
            center=self.halfway_line.center,
            radius=self._markings.mark_radius,
        )

    @property
    @abstractmethod
    def _penalty_arc_1(self) -> Circle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _penalty_area_1(self) -> Rectangle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _penalty_mark_1(self) -> Circle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _goal_area_1(self) -> Rectangle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    def _goal_1(self) -> Rectangle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    def _penalty_arc_2(self) -> Circle:
        """Right or top"""
        return self._penalty_arc_1.reflect(self.halfway_line)

    @property
    def _penalty_area_2(self) -> Rectangle:
        """Right or top"""
        return self._penalty_area_1.reflect(self.halfway_line)

    @property
    def _penalty_mark_2(self) -> Circle:
        """Right or top"""
        return self._penalty_mark_1.reflect(self.halfway_line)

    @property
    def _goal_area_2(self) -> Rectangle:
        """Right or top"""
        return self._goal_area_1.reflect(self.halfway_line)

    @property
    def _goal_2(self) -> Rectangle:
        """Right or top"""
        return self._goal_1.reflect(self.halfway_line)


class HorizontalPitch(Pitch):
    def _build_coord_sys(self) -> CoordinateSystem:
        x_dir = (
            "right"
            if self._touch_line_range[1] > self._touch_line_range[0]
            else "left"
        )
        y_dir = (
            "up"
            if self._goal_line_range[1] > self._goal_line_range[0]
            else "down"
        )
        return CoordinateSystem(origin=(0, 0), x_dir=x_dir, y_dir=y_dir)

    @property
    def bottom_left(self) -> Point:
        return Point(self._touch_line_range[0], self._goal_line_range[0])

    @property
    def bottom_right(self) -> Point:
        return Point(self._touch_line_range[1], self._goal_line_range[0])

    @property
    def top_left(self) -> Point:
        return Point(self._touch_line_range[0], self._goal_line_range[1])

    @property
    def top_right(self) -> Point:
        return Point(self._touch_line_range[1], self._goal_line_range[1])

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.bottom.center,
            end=self.top.center,
        )

    @property
    def _penalty_mark_point_1(self) -> Point:
        return Point(
            self._coord_sys.shift_x(
                self.bottom_left.x, self._markings.penalty_mark_distance
            ),
            self.left.center.y,
        )

    @property
    def _penalty_arc_1(self) -> Circle:
        return Circle(
            center=self._penalty_mark_point_1,
            radius=self._markings.centre_circle_radius,
        )

    @property
    def left_penalty_arc(self) -> Circle:
        return self._penalty_arc_1

    @property
    def _penalty_mark_1(self) -> Circle:
        return Circle(
            center=self._penalty_mark_point_1,
            radius=self._markings.mark_radius,
        )

    @property
    def left_penalty_mark(self) -> Circle:
        return self._penalty_mark_1

    @property
    def _penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def _goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def _goal_1(self) -> Rectangle:
        raise NotImplementedError

    def to_vertical(self) -> "VerticalPitch":
        raise NotImplementedError


class VerticalPitch(Pitch):
    def _build_coord_sys(self) -> CoordinateSystem:
        x_dir = (
            "right"
            if self._goal_line_range[1] > self._goal_line_range[0]
            else "left"
        )
        y_dir = (
            "up"
            if self._touch_line_range[1] > self._touch_line_range[0]
            else "down"
        )
        return CoordinateSystem(origin=(0, 0), x_dir=x_dir, y_dir=y_dir)

    @property
    def bottom_left(self) -> Point:
        return Point(self._goal_line_range[0], self._touch_line_range[0])

    @property
    def bottom_right(self) -> Point:
        return Point(self._goal_line_range[1], self._touch_line_range[0])

    @property
    def top_left(self) -> Point:
        return Point(self._goal_line_range[0], self._touch_line_range[1])

    @property
    def top_right(self) -> Point:
        return Point(self._goal_line_range[1], self._touch_line_range[1])

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.left.center,
            end=self.right.center,
        )

    @property
    def _penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    def _penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def _penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    def _goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def _goal_1(self) -> Rectangle:
        raise NotImplementedError

    def to_horizontal(self) -> HorizontalPitch:
        raise NotImplementedError
