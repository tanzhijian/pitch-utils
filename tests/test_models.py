from pitch_utils import MarkingDimensions, Markings


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
        assert markings.mode == "metric"
        assert markings.touch_line == 105
        assert markings.goal_line == 68
