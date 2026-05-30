from pitch_utils import MarkingDimensions, Markings


def multiply_by_100(num: float) -> int:
    return round(num * 100)


def test_marking_dimensions_scale() -> None:
    dims = MarkingDimensions()
    scaled_dims_1 = dims.scaled(0.5)
    scaled_dims_2 = dims * 0.5
    scaled_dims_3 = 0.5 * dims
    scaled_dims_4 = dims / 2
    assert scaled_dims_1 == scaled_dims_2 == scaled_dims_3 == scaled_dims_4


class TestMarkings:
    def test_default_params(self) -> None:
        markings = Markings()
        assert markings.mode == "standard"
        assert markings.touch_line == 105
        assert markings.goal_line == 68
        assert markings.penalty_mark_distance == 11
        assert markings.aspect_ratio() == 68 / 105

    def test_custom_spec_params(self) -> None:
        dims = MarkingDimensions(penalty_mark_distance=12)
        markings = Markings(spec=dims)
        assert markings.mode == "custom"
        assert markings.touch_line == 105
        assert markings.penalty_mark_distance == 12

    def test_scaled_spec_params(self) -> None:
        markings = Markings(touch_line=101, spec="scaled")
        assert markings.mode == "scaled"
        assert markings.touch_line == 101
        assert multiply_by_100(
            markings.penalty_mark_distance
        ) == multiply_by_100(11 * (101 / 105))

    def test_out_of_range_params(self) -> None:
        markings = Markings(touch_line=84)
        assert markings.mode == "scaled"
        assert markings.touch_line == 84
        assert multiply_by_100(
            markings.penalty_mark_distance
        ) == multiply_by_100(11 * 0.8)

    def test_within_range_params(self) -> None:
        markings = Markings(touch_line=101)
        assert markings.mode == "standard"
        assert markings.touch_line == 101
        assert markings.penalty_mark_distance == 11

    def test_out_of_range_and_custom_spec_params(self) -> None:
        dims = MarkingDimensions(penalty_mark_distance=12)
        markings = Markings(touch_line=120, spec=dims)
        assert markings.mode == "custom"
        assert markings.touch_line == 120
        assert markings.penalty_mark_distance == 12
