import pytest

from eradiate.contexts import MultiGenerator


@pytest.mark.parametrize(
    "input, expected",
    [
        ([range(5), range(10)], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([[0, 1, 2, 3, 0, 2, 1], [0, 1, 2, 3, 0, 2, 1, 4]], [0, 1, 2, 3, 4]),
        ([[3, 0, 2, 1], [0, 4, 2, 0, 1, 4]], [3, 0, 2, 1, 4]),
    ],
)
def test_multi_generator(input, expected):
    assert list(MultiGenerator(input)) == expected
