from abc import ABC, abstractmethod

from ._coordinate import Circle, CoordinateSystem, Line, Point, Rectangle
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

        self._markings = (
            markings
            if markings is not None
            else Markings(
                touch_line=self.touch_line.length,
                goal_line=self.goal_line.length,
            )
        )
        self._coor_sys = (
            coord_sys if coord_sys is not None else self._set_coor_sys()
        )

    @abstractmethod
    def _set_coor_sys(self) -> CoordinateSystem:
        raise NotImplementedError

    @property
    def coord_sys(self) -> CoordinateSystem:
        return self._coor_sys

    @property
    @abstractmethod
    def touch_line(self) -> Line:
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_line(self) -> Line:
        raise NotImplementedError

    @property
    def area(self) -> Rectangle:
        raise NotImplementedError

    @property
    def centre_circle(self) -> Circle:
        raise NotImplementedError

    @property
    def centre_mark(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def halfway_line(self) -> Line:
        raise NotImplementedError

    @property
    @abstractmethod
    def penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    @abstractmethod
    def penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def goal_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def penalty_arc_2(self) -> Circle:
        return self._coor_sys.reflect(self.penalty_arc_1, self.halfway_line)

    @property
    def penalty_area_2(self) -> Rectangle:
        return self._coor_sys.reflect(self.penalty_area_1, self.halfway_line)

    @property
    def penalty_mark_2(self) -> Circle:
        return self._coor_sys.reflect(self.penalty_mark_1, self.halfway_line)

    @property
    def goal_area_2(self) -> Rectangle:
        return self._coor_sys.reflect(self.goal_area_1, self.halfway_line)

    @property
    def goal_2(self) -> Rectangle:
        return self._coor_sys.reflect(self.goal_1, self.halfway_line)


class HorizontalPitch(Pitch):
    def _set_coor_sys(self) -> CoordinateSystem:
        return CoordinateSystem(
            x_range=self._touch_line_range,
            y_range=self._goal_line_range,
        )

    @property
    def touch_line(self) -> Line:
        return Line(
            start=Point(
                x=self._touch_line_range[0], y=self._goal_line_range[0]
            ),
            end=Point(x=self._touch_line_range[1], y=self._goal_line_range[0]),
        )

    @property
    def goal_line(self) -> Line:
        return Line(
            start=Point(
                x=self._touch_line_range[0], y=self._goal_line_range[0]
            ),
            end=Point(x=self._touch_line_range[0], y=self._goal_line_range[1]),
        )

    @property
    def halfway_line(self) -> Line:
        raise NotImplementedError

    @property
    def penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    def penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    def goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def goal_1(self) -> Rectangle:
        raise NotImplementedError


class VerticalPitch(Pitch):
    def _set_coor_sys(self) -> CoordinateSystem:
        return CoordinateSystem(
            x_range=self._goal_line_range,
            y_range=self._touch_line_range,
        )

    @property
    def touch_line(self) -> Line:
        return Line(
            start=Point(
                x=self._goal_line_range[0], y=self._touch_line_range[0]
            ),
            end=Point(x=self._goal_line_range[0], y=self._touch_line_range[1]),
        )

    @property
    def goal_line(self) -> Line:
        return Line(
            start=Point(
                x=self._goal_line_range[0], y=self._touch_line_range[0]
            ),
            end=Point(x=self._goal_line_range[1], y=self._touch_line_range[0]),
        )
