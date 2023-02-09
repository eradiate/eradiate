import pathlib

import numpy as np
import pytest

from eradiate.kernel import read_binary_grid3d, write_binary_grid3d


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 10),
        (3, 3, 4, 3),
        (4, 5, 1, 4),
    ],
)
def test_write_read_binary_grid3d(modes_all, shape, tmpdir):
    """Reads what was written."""

    length = np.prod(shape)
    write_values = np.random.random(length).reshape(shape)

    tmp_filename = pathlib.Path(tmpdir, "test.vol")
    write_binary_grid3d(filename=tmp_filename, values=write_values)
    read_values = read_binary_grid3d(tmp_filename)

    assert np.allclose(write_values, read_values)
