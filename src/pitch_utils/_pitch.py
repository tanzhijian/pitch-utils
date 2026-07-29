from abc import ABC, abstractmethod

from ._coordinates import (
    Circle,
    CoordinateSystem,
    DirectedRange,
    Line,
    Point,
    Rectangle,
)
from ._markings import Markings


class Pitch(ABC):
    def __init__(
        self,
        touch_line_range: DirectedRange,
        goal_line_range: DirectedRange,
        markings: Markings | None = None,
        coord_sys: CoordinateSystem | None = None,
    ) -> None:
        self._touch_line_range = touch_line_range
        self._goal_line_range = goal_line_range
        self._area = self._build_area()

        touch_line = touch_line_range.length
        goal_line = goal_line_range.length
        self._markings = (
            markings
            if markings is not None
            else Markings(touch_line=touch_line, goal_line=goal_line)
        )

        self._coord_sys = (
            coord_sys if coord_sys is not None else self._build_coord_sys()
        )

    @abstractmethod
    def _build_area(self) -> Rectangle:
        raise NotImplementedError

    @abstractmethod
    def _build_coord_sys(self) -> CoordinateSystem:
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if type(value) is not type(self):
            return False
        return (
            self.area == value.area
            and self.markings == value.markings
            and self.coord_sys == value.coord_sys
        )

    @property
    def touch_line_range(self) -> DirectedRange:
        return self._touch_line_range

    @property
    def goal_line_range(self) -> DirectedRange:
        return self._goal_line_range

    @property
    def area(self) -> Rectangle:
        return self._area

    @property
    def coord_sys(self) -> CoordinateSystem:
        return self._coord_sys

    @property
    def markings(self) -> Markings:
        return self._markings

    @property
    def _left_x(self) -> float:
        min_x = self._area.min_point.x
        max_x = self._area.max_point.x
        return min_x if self._coord_sys.x_dir == "right" else max_x

    @property
    def _right_x(self) -> float:
        min_x = self._area.min_point.x
        max_x = self._area.max_point.x
        return max_x if self._coord_sys.x_dir == "right" else min_x

    @property
    def _bottom_y(self) -> float:
        min_y = self._area.min_point.y
        max_y = self._area.max_point.y
        return min_y if self._coord_sys.y_dir == "up" else max_y

    @property
    def _top_y(self) -> float:
        min_y = self._area.min_point.y
        max_y = self._area.max_point.y
        return max_y if self._coord_sys.y_dir == "up" else min_y

    @property
    def bottom_left(self) -> Point:
        return Point(self._left_x, self._bottom_y)

    @property
    def bottom_right(self) -> Point:
        return Point(self._right_x, self._bottom_y)

    @property
    def top_left(self) -> Point:
        return Point(self._left_x, self._top_y)

    @property
    def top_right(self) -> Point:
        return Point(self._right_x, self._top_y)

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
            center=self._area.center,
            radius=self._markings.centre_circle_radius,
        )

    @property
    def centre_mark(self) -> Circle:
        return Circle(
            center=self._area.center,
            radius=self._markings.mark_radius,
        )

    @property
    @abstractmethod
    def halfway_line(self) -> Line:
        raise NotImplementedError

    @property
    @abstractmethod
    def _penalty_mark_1_point(self) -> Point:
        """Left or bottom"""
        raise NotImplementedError

    @property
    def _penalty_arc_1(self) -> Circle:
        """Left or bottom"""
        return Circle(
            center=self._penalty_mark_1_point,
            radius=self._markings.centre_circle_radius,
        )

    @property
    def _penalty_mark_1(self) -> Circle:
        """Left or bottom"""
        return Circle(
            center=self._penalty_mark_1_point,
            radius=self._markings.mark_radius,
        )

    @property
    @abstractmethod
    def _penalty_area_1(self) -> Rectangle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _goal_area_1(self) -> Rectangle:
        """Left or bottom"""
        raise NotImplementedError

    @property
    @abstractmethod
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

    @abstractmethod
    def transpose(self) -> "Pitch":
        raise NotImplementedError


class HorizontalPitch(Pitch):
    def _build_area(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                self._touch_line_range.start, self._goal_line_range.start
            ),
            p2=Point(self._touch_line_range.end, self._goal_line_range.end),
        )

    def _build_coord_sys(self) -> CoordinateSystem:

        return CoordinateSystem(
            origin=Point(x=0, y=0),
            x_range=self._touch_line_range,
            y_range=self._goal_line_range,
        )

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.bottom.center,
            end=self.top.center,
        )

    @property
    def _penalty_mark_1_point(self) -> Point:
        return Point(
            x=self._coord_sys.shift_x(
                self.bottom_left.x, self._markings.penalty_mark_distance
            ),
            y=self.left.center.y,
        )

    @property
    def left_penalty_arc(self) -> Circle:
        return self._penalty_arc_1

    @property
    def left_penalty_mark(self) -> Circle:
        return self._penalty_mark_1

    @property
    def _penalty_area_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self.bottom_left.x,
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                    self._markings.penalty_area_length,
                    op="-",
                ),
            ),
            p2=Point(
                self.bottom_left.x + self._markings.penalty_area_length,
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                    self._markings.penalty_area_length,
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
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                    self._markings.goal_area_length,
                    op="-",
                ),
            ),
            p2=Point(
                x=self.bottom_left.x + self._markings.goal_area_length,
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                    self._markings.goal_area_length,
                ),
            ),
        )

    @property
    def left_goal_area(self) -> Rectangle:
        return self._goal_area_1

    @property
    def _goal_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self._coord_sys.shift_x(
                    self.bottom_left.x,
                    self._markings.goal_height,
                    op="-",
                ),
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                    op="-",
                ),
            ),
            p2=Point(
                x=self.bottom_left.x,
                y=self._coord_sys.shift_y(
                    self.left.center.y,
                    self._markings.goal_width / 2,
                ),
            ),
        )

    @property
    def left_goal(self) -> Rectangle:
        return self._goal_1

    @property
    def right_penalty_arc(self) -> Circle:
        return self._penalty_arc_2

    @property
    def right_penalty_area(self) -> Rectangle:
        return self._penalty_area_2

    @property
    def right_penalty_mark(self) -> Circle:
        return self._penalty_mark_2

    @property
    def right_goal_area(self) -> Rectangle:
        return self._goal_area_2

    @property
    def right_goal(self) -> Rectangle:
        return self._goal_2

    def transpose(self) -> "VerticalPitch":
        raise NotImplementedError


class VerticalPitch(Pitch):
    def _build_area(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                self._goal_line_range.start, self._touch_line_range.start
            ),
            p2=Point(self._goal_line_range.end, self._touch_line_range.end),
        )

    def _build_coord_sys(self) -> CoordinateSystem:
        return CoordinateSystem(
            origin=Point(x=0, y=0),
            x_range=self._goal_line_range,
            y_range=self._touch_line_range,
        )

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.left.center,
            end=self.right.center,
        )

    @property
    def _penalty_mark_1_point(self) -> Point:
        return Point(
            x=self.bottom.center.x,
            y=self._coord_sys.shift_y(
                self.bottom_left.y, self._markings.penalty_mark_distance
            ),
        )

    @property
    def bottom_penalty_arc(self) -> Circle:
        return self._penalty_arc_1

    @property
    def bottom_penalty_mark(self) -> Circle:
        return self._penalty_mark_1

    @property
    def _penalty_area_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                    self._markings.penalty_area_length,
                    op="-",
                ),
                y=self.bottom_left.y,
            ),
            p2=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                    self._markings.penalty_area_length,
                ),
                y=self.bottom_left.y + self._markings.penalty_area_length,
            ),
        )

    @property
    def bottom_penalty_area(self) -> Rectangle:
        return self._penalty_area_1

    @property
    def _goal_area_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                    self._markings.goal_area_length,
                    op="-",
                ),
                y=self.bottom_left.y,
            ),
            p2=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                    self._markings.goal_area_length,
                ),
                y=self.bottom_left.y + self._markings.goal_area_length,
            ),
        )

    @property
    def bottom_goal_area(self) -> Rectangle:
        return self._goal_area_1

    @property
    def _goal_1(self) -> Rectangle:
        return Rectangle(
            p1=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                    op="-",
                ),
                y=self._coord_sys.shift_y(
                    self.bottom_left.y,
                    self._markings.goal_height,
                    op="-",
                ),
            ),
            p2=Point(
                x=self._coord_sys.shift_x(
                    self.bottom.center.x,
                    self._markings.goal_width / 2,
                ),
                y=self.bottom_left.y,
            ),
        )

    @property
    def bottom_goal(self) -> Rectangle:
        return self._goal_1

    @property
    def top_penalty_arc(self) -> Circle:
        return self._penalty_arc_2

    @property
    def top_penalty_mark(self) -> Circle:
        return self._penalty_mark_2

    @property
    def top_penalty_area(self) -> Rectangle:
        return self._penalty_area_2

    @property
    def top_goal_area(self) -> Rectangle:
        return self._goal_area_2

    @property
    def top_goal(self) -> Rectangle:
        return self._goal_2

    def transpose(self) -> HorizontalPitch:
        raise NotImplementedError
