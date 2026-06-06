from abc import ABC, abstractmethod

from ._coordinates import Circle, CoordinateSystem, Line, Point, Rectangle
from ._markings import Markings


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
        return self._coord_sys.reflect(self._penalty_arc_1, self.halfway_line)

    @property
    def _penalty_area_2(self) -> Rectangle:
        """Right or top"""
        return self._coord_sys.reflect(self._penalty_area_1, self.halfway_line)

    @property
    def _penalty_mark_2(self) -> Circle:
        """Right or top"""
        return self._coord_sys.reflect(self._penalty_mark_1, self.halfway_line)

    @property
    def _goal_area_2(self) -> Rectangle:
        """Right or top"""
        return self._coord_sys.reflect(self._goal_area_1, self.halfway_line)

    @property
    def _goal_2(self) -> Rectangle:
        """Right or top"""
        return self._coord_sys.reflect(self._goal_1, self.halfway_line)


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
        return Rectangle(
            p1=Point(
                x=self.bottom_left.x,
                y=(
                    self.left.center.y
                    - self._markings.goal_width / 2
                    - self._markings.penalty_area_length
                ),
            ),
            p2=Point(
                self.bottom_left.x + self._markings.penalty_area_length,
                y=(
                    self.left.center.y
                    + self._markings.goal_width / 2
                    + self._markings.penalty_area_length
                ),
            ),
        )

    @property
    def left_penalty_area(self) -> Rectangle:
        return self._penalty_area_1

    @property
    def _goal_area_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self.bottom_left.x,
                y=(
                    self.left.center.y
                    - self._markings.goal_width / 2
                    - self._markings.goal_area_length
                ),
            ),
            p2=Point(
                x=self.bottom_left.x + self._markings.goal_area_length,
                y=(
                    self.left.center.y
                    + self._markings.goal_width / 2
                    + self._markings.goal_area_length
                ),
            ),
        )

    @property
    def left_goal_area(self) -> Rectangle:
        return self._goal_area_1

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
