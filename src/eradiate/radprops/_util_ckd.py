import pandas as pd

from .. import data

PATHS = {
    "afgl_1986-us_standard-10nm": "ckd/absorption/10nm/afgl_1986-us_standard-10nm.nc",
    "afgl_1986-us_standard-1nm": "ckd/absorption/1nm/afgl_1986-us_standard-1nm.nc",
}


def open_dataset(id):
    # Single-file version
    ds = data.open_dataset(PATHS[id])

    # Combine the 'bin' and 'index' coordinates into a multi-index, then reindex dataset
    idx = pd.MultiIndex.from_arrays(
        (ds.bin.values, ds.index.values), names=("bin", "index")
    )
    ds = ds.drop_vars(("bin", "index"))
    ds = ds.reindex({"bd": idx})

    return ds
