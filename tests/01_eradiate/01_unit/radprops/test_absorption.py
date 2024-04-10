"""
Tests for the radprops._absorption module.
"""

import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.radprops import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)

# ------------------------------------------------------------------------------
#                                    Fixtures
# ------------------------------------------------------------------------------


@pytest.mark.usefixtures("mode_mono")
class TestMonoAbsorptionDatabase:
    @pytest.fixture(scope="class")
    def db(self):
        eradiate.set_mode("none")
        _db = MonoAbsorptionDatabase.from_name("komodo")
        yield _db
        _db.cache_clear()

    def test_construct(self, db):
        # Default komodo settings use lazy data loading
        assert db.lazy is True

        # The dict converter accepts kwargs and can be used to override defaults
        db = MonoAbsorptionDatabase.from_dict(
            {"construct": "from_name", "name": "komodo", "lazy": False}
        )
        assert db.lazy is False

    @pytest.mark.parametrize(
        "w",
        [
            np.array([550.0]) * ureg.nm,
            np.linspace(540.0, 560.0) * ureg.nm,
        ],
    )
    def test_eval(
        self, db, thermoprops_us_standard, absorption_database_error_handler_config, w
    ):
        sigma_a = db.eval_sigma_a_mono(
            w,
            thermoprops_us_standard,
            ErrorHandlingConfiguration.convert(
                absorption_database_error_handler_config
            ),
        )

        # sigma_a should have a shape of (w, z)
        z = thermoprops_us_standard.z.values
        assert sigma_a.values.shape == (w.size, z.size)


@pytest.mark.usefixtures("mode_ckd")
class TestCKDAbsorptionDatabase:
    @pytest.fixture(scope="class")
    def db(self):
        eradiate.set_mode("none")
        _db = CKDAbsorptionDatabase.from_name("monotropa")
        yield _db
        _db.cache_clear()

    def test_construct(self, db):
        # Default monotropa settings use eager data loading
        assert db.lazy is False

        # The dict converter accepts kwargs and can be used to override defaults
        db = MonoAbsorptionDatabase.from_dict(
            {"construct": "from_name", "name": "monotropa", "lazy": True}
        )
        assert db.lazy is True

    @pytest.mark.parametrize(
        "w, expected",
        [
            ({"wl": 550.0}, ["monotropa-18100_18200.nc"]),
            ({"wl": 550.0 * ureg.nm}, ["monotropa-18100_18200.nc"]),
            ({"wl": 0.55 * ureg.micron}, ["monotropa-18100_18200.nc"]),
            ({"wl": [550.0, 550.0]}, ["monotropa-18100_18200.nc"] * 2),
        ],
        ids=[
            "wl_scalar_unitless",
            "wl_scalar_nm",
            "wl_scalar_micron",
            "wl_array_unitless",
        ],
    )
    def test_filename_lookup(self, db, w, expected):
        assert db.lookup_filenames(**w) == expected

    @pytest.mark.parametrize(
        "wg",
        [([550.0] * ureg.nm, 0.5)],
    )
    def test_eval(
        self, db, thermoprops_us_standard, absorption_database_error_handler_config, wg
    ):
        sigma_a = db.eval_sigma_a_ckd(
            *wg,
            thermoprops_us_standard,
            ErrorHandlingConfiguration.convert(
                absorption_database_error_handler_config
            ),
        )

        # sigma_a should have a shape of (w, z)
        z = thermoprops_us_standard.z.values
        assert sigma_a.values.shape == (wg[0].size, z.size)

    def test_cache_clear(self, db):
        # Make a query to ensure that the cache is filling up
        db.load_dataset("monotropa-18100_18200.nc")
        assert db._cache.currsize > 0
        # Clear the cache: it should be empty after that
        db.cache_clear()
        assert db._cache.currsize == 0

    def test_cache_reset(self, db):
        db.cache_reset(2)
        assert db._cache.currsize == 0
        assert db._cache.maxsize == 2
        db.cache_reset(8)
        assert db._cache.currsize == 0
        assert db._cache.maxsize == 8
