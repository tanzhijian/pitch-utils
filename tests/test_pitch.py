import pytest

from pitch_utils import HorizontalPitch, Line, Point, Rectangle, VerticalPitch


class TestLine:
    @pytest.fixture(scope="class")
    def line(self) -> Line:
        return Line(Point(0, 0), Point(100, 0))

    def test_length(self, line: Line) -> None:
        assert line.length == 100

    def test_center(self, line: Line) -> None:
        center = line.center
        assert center.x == 50
        assert center.y == 0


class TestRectangle:
    @pytest.fixture(scope="class")
    def rectangle(self) -> Rectangle:
        return Rectangle(bottom_left=Point(0, 0), width=100, height=60)

    def test_bottom_right(self, rectangle: Rectangle) -> None:
        bottom_right = rectangle.bottom_right
        assert bottom_right.x == 100
        assert bottom_right.y == 0

    def test_top_left(self, rectangle: Rectangle) -> None:
        top_left = rectangle.top_left
        assert top_left.x == 0
        assert top_left.y == 60

    def test_top_right(self, rectangle: Rectangle) -> None:
        top_right = rectangle.top_right
        assert top_right.x == 100
        assert top_right.y == 60

    def test_left(self, rectangle: Rectangle) -> None:
        left = rectangle.left
        assert left.start == Point(0, 0)
        assert left.end == Point(0, 60)

    def test_right(self, rectangle: Rectangle) -> None:
        right = rectangle.right
        assert right.start == Point(100, 0)
        assert right.end == Point(100, 60)

    def test_top(self, rectangle: Rectangle) -> None:
        top = rectangle.top
        assert top.start == Point(0, 60)
        assert top.end == Point(100, 60)

    def test_bottom(self, rectangle: Rectangle) -> None:
        bottom = rectangle.bottom
        assert bottom.start == Point(0, 0)
        assert bottom.end == Point(100, 0)


class TestHorizontalPitch:
    @pytest.fixture(scope="class")
    def pitch(self) -> HorizontalPitch:
        return HorizontalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )

    def test_coord_sys(self, pitch: HorizontalPitch) -> None:
        coord_sys = pitch.coord_sys
        assert coord_sys.x_dir == "right"
        assert coord_sys.y_dir == "up"
        assert coord_sys.origin == (0, 0)

    def test_touch_line(self, pitch: HorizontalPitch) -> None:
        touch_line = pitch.touch_line
        assert touch_line.start == Point(0, 0)
        assert touch_line.end == Point(105, 0)

    def test_goal_line(self, pitch: HorizontalPitch) -> None:
        goal_line = pitch.goal_line
        assert goal_line.start == Point(0, 0)
        assert goal_line.end == Point(0, 68)

    def test_canvas(self, pitch: HorizontalPitch) -> None:
        canvas = pitch.canvas
        assert canvas.bottom_left == Point(0, 0)
        assert canvas.width == 105
        assert canvas.height == 68

    def test_centre_circle(self, pitch: HorizontalPitch) -> None:
        centre_circle = pitch.centre_circle
        assert centre_circle.center == Point(52.5, 34)
        assert centre_circle.radius == 9.15

    def test_halfway_line(self, pitch: HorizontalPitch) -> None:
        halfway_line = pitch.halfway_line
        assert halfway_line.start == Point(52.5, 0)
        assert halfway_line.end == Point(52.5, 68)

    def test_centre_mark(self, pitch: HorizontalPitch) -> None:
        centre_mark = pitch.centre_mark
        assert centre_mark.center == Point(52.5, 34)
        assert centre_mark.radius == 0.1


class TestVerticalPitch:
    @pytest.fixture(scope="class")
    def pitch(self) -> VerticalPitch:
        return VerticalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )

    def test_coord_sys(self, pitch: VerticalPitch) -> None:
        coord_sys = pitch.coord_sys
        assert coord_sys.x_dir == "right"
        assert coord_sys.y_dir == "up"
        assert coord_sys.origin == (0, 0)

    def test_touch_line(self, pitch: VerticalPitch) -> None:
        touch_line = pitch.touch_line
        assert touch_line.start == Point(0, 0)
        assert touch_line.end == Point(0, 105)

    def goal_line(self, pitch: VerticalPitch) -> None:
        goal_line = pitch.goal_line
        assert goal_line.start == Point(0, 0)
        assert goal_line.end == Point(68, 0)

    def test_canvas(self, pitch: VerticalPitch) -> None:
        canvas = pitch.canvas
        assert canvas.bottom_left == Point(0, 0)
        assert canvas.width == 68
        assert canvas.height == 105

    def test_halfway_line(self, pitch: VerticalPitch) -> None:
        halfway_line = pitch.halfway_line
        assert halfway_line.start == Point(0, 52.5)
        assert halfway_line.end == Point(68, 52.5)
