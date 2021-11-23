import pytest

from eradiate import data
from eradiate import unit_registry as ureg
from eradiate.ckd import Bin, BinSet, bin_filter_ids, bin_filter_interval
from eradiate.quad import Quad


def test_ckd_bin(mode_ckd):
    """
    Unit tests for :class:`eradiate.ckd.Bin`.
    """
    quad = Quad.gauss_legendre(16)

    # Construct a bin
    bin = Bin(id="505", wmin=500.0, wmax=510.0, quad=quad)

    # Scalar values are correctly converted to config default units
    assert bin.wmin == 500.0 * ureg.nm
    assert bin.wmax == 510.0 * ureg.nm

    # Central wavelength is the mean of bounds
    assert bin.wcenter == 505.0 * ureg.nm

    # Width is the difference between bounds
    assert bin.width == 10.0 * ureg.nm

    # Wrong bound ordering raises
    with pytest.raises(ValueError):
        Bin(id="505", wmin=510.0, wmax=500.0, quad=quad)

    # Conversion from tuple is supported
    assert isinstance(Bin.convert(("505", 500.0, 510.0, quad)), Bin)

    # Conversion from dict is supported
    assert isinstance(
        Bin.convert(dict(id="505", wmin=500.0, wmax=510.0, quad=quad)), Bin
    )


def test_ckd_bin_filter_interval(mode_ckd):
    """
    Unit tests for :func:`eradiate.ckd.bin_filter_interval`.
    """
    quad = Quad.gauss_legendre(16)

    assert bin_filter_interval(wmin=505 * ureg.nm, wmax=595 * ureg.nm)(
        Bin(id="foo", wmin=510, wmax=520, quad=quad)
    )
    assert not bin_filter_interval(wmin=505 * ureg.nm, wmax=595 * ureg.nm)(
        Bin(id="foo", wmin=400, wmax=410, quad=quad)
    )

    # By default, coverage is such that enpoints are included
    assert bin_filter_interval(wmin=505 * ureg.nm, wmax=595 * ureg.nm)(
        Bin(id="foo", wmin=500, wmax=510, quad=quad)
    )
    assert bin_filter_interval(wmin=505 * ureg.nm, wmax=595 * ureg.nm)(
        Bin(id="foo", wmin=590, wmax=600, quad=quad)
    )

    # Endpoint exclusion flag trims endpoint bins
    assert not bin_filter_interval(
        wmin=505 * ureg.nm, wmax=595 * ureg.nm, endpoints=False
    )(Bin(id="foo", wmin=500, wmax=510, quad=quad))
    assert not bin_filter_interval(
        wmin=505 * ureg.nm, wmax=595 * ureg.nm, endpoints=False
    )(Bin(id="foo", wmin=590, wmax=600, quad=quad))

    # Tightly fit bounds pass the filter
    assert bin_filter_interval(wmin=500 * ureg.nm, wmax=510 * ureg.nm, endpoints=True)(
        Bin(id="foo", wmin=500, wmax=510, quad=quad)
    )

    # wmin == wmax selects a bin which contains wmin
    assert bin_filter_interval(wmin=505 * ureg.nm, wmax=505 * ureg.nm, endpoints=True)(
        Bin(id="foo", wmin=500, wmax=510, quad=quad)
    )


def test_ckd_bin_set_constructors(mode_ckd):
    """
    Unit tests for :class:`eradiate.ckd.BinSet` constructors.
    """

    # Load from database
    bin_set = BinSet.from_db("10nm")
    assert bin_set.id == "10nm"

    # Load from node (i.e. gas absorption) dataset
    ds = data.open("ckd_absorption", "afgl_1986-us_standard-10nm_test")
    ds.load()
    ds.close()

    bin_set = BinSet.from_node_dataset(ds)
    assert bin_set.id == "10nm_test"


def test_ckd_bin_set_filter(mode_ckd):
    """
    Unit tests for :meth:`eradiate.ckd.BinSet.filter_bins`.
    """
    bin_set = BinSet.from_db("10nm_test")

    # We get a single bin using wmin == wmax
    filter = bin_filter_interval(wmin=550 * ureg.nm, wmax=550 * ureg.nm, endpoints=True)
    assert len(bin_set.filter_bins(filter)) == 1

    # We extend the range to get 2 items
    filter = bin_filter_interval(wmin=545 * ureg.nm, wmax=565 * ureg.nm, endpoints=True)
    assert len(bin_set.filter_bins(filter)) == 2

    # The previous also works with endpoints set to False
    filter = bin_filter_interval(
        wmin=545 * ureg.nm, wmax=565 * ureg.nm, endpoints=False
    )
    assert len(bin_set.filter_bins(filter)) == 2

    # Now we further extend the range to get 3 bins
    filter = bin_filter_interval(wmin=545 * ureg.nm, wmax=570 * ureg.nm, endpoints=True)
    assert len(bin_set.filter_bins(filter)) == 3

    # Setting endpoints to False excludes partly covered bins
    filter = bin_filter_interval(
        wmin=545 * ureg.nm, wmax=570 * ureg.nm, endpoints=False
    )
    assert len(bin_set.filter_bins(filter)) == 2

    # Multiple filters can be combined
    assert (
        len(
            bin_set.filter_bins(
                bin_filter_interval(wmin=545 * ureg.nm, wmax=555 * ureg.nm),
                bin_filter_ids(["510"]),
            )
        )
        == 2
    )

    # A bin can only be selected once
    assert (
        len(
            bin_set.filter_bins(
                bin_filter_interval(wmin=545 * ureg.nm, wmax=555 * ureg.nm),
                bin_filter_ids(["550"]),
            )
        )
        == 1
    )
    assert (
        len(
            bin_set.filter_bins(
                bin_filter_interval(wmin=545 * ureg.nm, wmax=565 * ureg.nm),
                bin_filter_ids(["550", "560"]),
            )
        )
        == 2
    )


def test_ckd_bin_set_select(mode_ckd):
    """
    Unit tests for :meth:`eradiate.ckd.BinSet.select_bins`.
    """

    bin_set = BinSet.from_db("10nm_test")

    # Select with an explicit filter
    filter = bin_filter_interval(wmin=550 * ureg.nm, wmax=560 * ureg.nm)
    assert len(bin_set.select_bins(filter)) == 2

    # Select with a string
    assert len(bin_set.select_bins("550", "560")) == 2

    # Select with a tuple
    assert (
        len(
            bin_set.select_bins(
                ("interval", {"wmin": 550 * ureg.nm, "wmax": 560 * ureg.nm})
            )
        )
        == 2
    )

    # Select with a dict
    assert (
        len(
            bin_set.select_bins(
                {
                    "type": "interval",
                    "filter_kwargs": {"wmin": 550 * ureg.nm, "wmax": 560 * ureg.nm},
                }
            )
        )
        == 2
    )
