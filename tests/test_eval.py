from src.eval import _percentile


def test_percentile_median():
    assert _percentile([1, 2, 3, 4, 5], 0.5) == 3


def test_percentile_p95_of_single_value():
    assert _percentile([7.0], 0.95) == 7.0


def test_percentile_clamps_to_max():
    assert _percentile([1, 2, 3], 1.0) == 3
