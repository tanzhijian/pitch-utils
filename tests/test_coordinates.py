from typing import Literal, cast

import numpy as np
import pytest

from pitch_utils import (
    Circle,
    CoordinateSystem,
    DirectedRange,
    Line,
    Point,
    Points,
    Rectangle,
)


class TestPoints:
    @pytest.fixture(scope="class")
    def points(self) -> Points:
        return Points.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )

    def test_init_rejects_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            Points(np.array([1.0, 2.0]))

    def test_from_points(self, points: Points) -> None:
        point_list = [Point(1, 2), Point(3, 4), Point(5, 6)]
        points_2 = Points.from_points(point_list)
        assert points_2.to_list() == points.to_list()

    def test_from_rows(self, points: Points) -> None:
        rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        points_2 = Points.from_rows(rows)
        assert points_2.to_list() == points.to_list()

    def test_from_numpy(self, points: Points) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        points_2 = Points.from_numpy(arr)
        assert points_2.to_list() == points.to_list()

    def test_to_list(self, points: Points) -> None:
        expected = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assert points.to_list() == expected

    def test_to_points(self, points: Points) -> None:
        expected = [Point(1, 2), Point(3, 4), Point(5, 6)]
        assert points.to_points() == expected

    def test_to_numpy(self, points: Points) -> None:
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(points.to_numpy(), expected)

    def test_iter_tuples(self, points: Points) -> None:
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        assert list(points.iter_tuples()) == expected


class TestPoint:
    @pytest.fixture(scope="class")
    def point(self) -> Point:
        return Point(1, 1)

    def test_eq(self, point: Point) -> None:
        assert point == Point(1, 1)
        assert point != Point(2, 2)
        assert point != (1, 1)
        assert Point(x=0.1 + 0.2, y=1) == Point(x=0.3, y=1)

    def test_lt(self, point: Point) -> None:
        assert point < Point(2, 1)
        assert point < Point(1, 2)
        with pytest.raises(TypeError):
            assert point < 2

    def test_coords(self, point: Point) -> None:
        assert point.coords == (1, 1)


class TestLine:
    @pytest.fixture(scope="class")
    def line(self) -> Line:
        return Line(Point(0, 0), Point(100, 0))

    def test_eq(self, line: Line) -> None:
        assert line == Line(Point(0, 0), Point(100, 0))
        assert line == Line(Point(100, 0), Point(0, 0))
        assert line != Line(Point(0, 0), Point(50, 0))
        assert line != 2

    def test_strictly_equal(self, line: Line) -> None:
        assert line.is_strictly_equal(Line(Point(0, 0), Point(100, 0)))
        assert not line.is_strictly_equal(Line(Point(100, 0), Point(0, 0)))

    def test_length(self, line: Line) -> None:
        assert line.length == 100

    def test_center(self, line: Line) -> None:
        center = line.center
        assert center.x == 50
        assert center.y == 0

    def test_coords(self, line: Line) -> None:
        assert line.coords == ((0, 0), (100, 0))


class TestCircle:
    @pytest.fixture(scope="class")
    def circle(self) -> Circle:
        return Circle(Point(0, 0), 10)

    def test_eq(self, circle: Circle) -> None:
        assert circle == Circle(Point(0, 0), 10)
        assert circle != Circle(Point(1, 1), 10)
        assert circle != Circle(Point(0, 0), 5)
        assert circle != 2


class TestRectangle:
    @pytest.fixture(scope="class")
    def rectangle(self) -> Rectangle:
        return Rectangle(p1=Point(0, 0), p2=Point(100, 60))

    def test_eq(self, rectangle: Rectangle) -> None:
        assert rectangle == Rectangle(Point(0, 0), Point(100, 60))
        assert rectangle != Rectangle(Point(1, 0), Point(100, 60))
        assert rectangle != 2

    def test_coords(self, rectangle: Rectangle) -> None:
        assert rectangle.coords == ((0, 0), (0, 60), (100, 0), (100, 60))

    def test_center(self, rectangle: Rectangle) -> None:
        center = rectangle.center
        assert center.x == 50
        assert center.y == 30


class TestDirectedRange:
    @pytest.fixture(scope="class")
    def drange(self) -> DirectedRange:
        return DirectedRange(0, 100)

    def test_coords(self, drange: DirectedRange) -> None:
        assert drange.coords == (0, 100)

    def test_length(self, drange: DirectedRange) -> None:
        assert drange.length == 100


class TestDefaultCoordSys:
    @pytest.fixture(scope="class")
    def cs(self) -> CoordinateSystem:
        return CoordinateSystem(
            origin=Point(0, 0),
            x_range=DirectedRange(0, 105),
            y_range=DirectedRange(0, 68),
        )

    def test_eq(self, cs: CoordinateSystem) -> None:
        assert cs == CoordinateSystem(
            origin=Point(0, 0),
            x_range=DirectedRange(0, 105),
            y_range=DirectedRange(0, 68),
        )
        assert cs != CoordinateSystem(
            origin=Point(1, 0),
            x_range=DirectedRange(0, 105),
            y_range=DirectedRange(0, 68),
        )
        assert cs != CoordinateSystem(
            origin=Point(0, 0),
            x_range=DirectedRange(105, 0),
            y_range=DirectedRange(0, 68),
        )
        assert cs != CoordinateSystem(
            origin=Point(0, 0),
            x_range=DirectedRange(0, 105),
            y_range=DirectedRange(68, 0),
        )
        assert cs != 2

    def test_shift_point(self, cs: CoordinateSystem) -> None:
        assert cs.shift(Point(1, 2), x_offset=10) == Point(11, 2)
        assert cs.shift(
            Point(1, 2), x_offset=10, y_offset=3, x_op="-"
        ) == Point(-9, 5)

    def test_shift_line(self, cs: CoordinateSystem) -> None:
        shifted = cs.shift(
            Line(Point(1, 2), Point(3, 4)), y_offset=5, y_op="-"
        )
        assert shifted.is_strictly_equal(Line(Point(1, -3), Point(3, -1)))

    def test_shift_circle(self, cs: CoordinateSystem) -> None:
        assert cs.shift(Circle(Point(2, 3), 4), x_offset=5) == Circle(
            Point(7, 3), 4
        )

    def test_shift_rectangle(self, cs: CoordinateSystem) -> None:
        assert cs.shift(
            Rectangle(Point(1, 2), Point(3, 4)),
            x_offset=5,
            y_offset=6,
            y_op="-",
        ) == Rectangle(Point(6, -4), Point(8, -2))

    def test_shift_rejects_invalid_operation(
        self, cs: CoordinateSystem
    ) -> None:
        invalid_op = cast(Literal["+", "-"], "invalid")
        with pytest.raises(ValueError, match=r"Operation must be '\+' or '-'"):
            cs.shift(Point(1, 2), x_op=invalid_op)

    def test_reflect_point(self, cs: CoordinateSystem) -> None:
        pivot = Line(Point(0, -1), Point(0, 1))
        assert cs.reflect(Point(1, 2), pivot) == Point(-1, 2)

    def test_reflect_line(self, cs: CoordinateSystem) -> None:
        pivot = Line(Point(0, -1), Point(0, 1))
        reflected = cs.reflect(Line(Point(1, 0), Point(3, 0)), pivot)
        assert reflected.is_strictly_equal(Line(Point(-1, 0), Point(-3, 0)))

    def test_reflect_circle(self, cs: CoordinateSystem) -> None:
        pivot = Line(Point(0, -1), Point(0, 1))
        assert cs.reflect(Circle(Point(2, 0), 3), pivot) == Circle(
            Point(-2, 0), 3
        )

    def test_reflect_rectangle(self, cs: CoordinateSystem) -> None:
        pivot = Line(Point(0, -1), Point(0, 1))
        assert cs.reflect(
            Rectangle(Point(1, 1), Point(3, 2)), pivot
        ) == Rectangle(Point(-3, 1), Point(-1, 2))

    def test_rotate_point(self, cs: CoordinateSystem) -> None:
        assert cs.rotate(Point(1, 0), 90, Point(0, 0)) == Point(0, 1)
        assert cs.rotate(Point(-1, 0), 180, Point(0, 0)) == Point(1, 0)

    def test_rotate_line(self, cs: CoordinateSystem) -> None:
        rotated = cs.rotate(Line(Point(1, 0), Point(3, 0)), 90, Point(0, 0))
        assert rotated.is_strictly_equal(Line(Point(0, 1), Point(0, 3)))

    def test_rotate_circle(self, cs: CoordinateSystem) -> None:
        assert cs.rotate(Circle(Point(2, 0), 3), 90, Point(0, 0)) == Circle(
            Point(0, 2), 3
        )

    def test_rotate_rectangle(self, cs: CoordinateSystem) -> None:
        assert cs.rotate(
            Rectangle(Point(1, 1), Point(3, 2)), 90, Point(0, 0)
        ) == Rectangle(Point(-2, 1), Point(-1, 3))
        assert cs.rotate(
            Rectangle(Point(0, 13.84), Point(16.5, 54.16)),
            180,
            Point(52.5, 34),
        ) == Rectangle(Point(88.5, 13.84), Point(105, 54.16))


class TestLeftDownCoordSys:
    @pytest.fixture
    def cs(self) -> CoordinateSystem:
        return CoordinateSystem(
            origin=Point(0, 0),
            x_range=DirectedRange(105, 0),
            y_range=DirectedRange(68, 0),
        )

    def test_shift_point(self, cs: CoordinateSystem) -> None:
        assert cs.shift(Point(1, 2), x_offset=10, y_offset=3) == Point(-9, -1)
        assert cs.shift(
            Point(1, 2), x_offset=10, y_offset=3, x_op="-", y_op="-"
        ) == Point(11, 5)

    def test_reflect_point(self, cs: CoordinateSystem) -> None:
        pivot = Line(Point(0, -1), Point(0, 1))
        assert cs.reflect(Point(1, 2), pivot) == Point(-1, 2)

    def test_rotate_point(self, cs: CoordinateSystem) -> None:
        assert cs.rotate(Point(1, 0), 90, Point(0, 0)) == Point(0, 1)
        assert cs.rotate(Point(-1, 0), 180, Point(0, 0)) == Point(1, 0)
