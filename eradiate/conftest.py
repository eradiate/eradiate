import pytest

import eradiate


# Create kernel variant fixtures
def generate_fixture(variant):
    @pytest.fixture()
    def fixture():
        try:
            import eradiate.kernel
            eradiate.kernel.set_variant(variant)
        except Exception:
            pytest.skip(f"Mitsuba variant '{variant}' is not enabled!")

    globals()["variant_" + variant] = fixture


for variant in ["scalar_mono", "scalar_mono_double",
                "scalar_mono_polarized",
                "scalar_rgb", "scalar_spectral",
                "packet_rgb", "packet_spectral"]:
    generate_fixture(variant)
del generate_fixture


# Create mode fixtures
def generate_fixture(mode):
    @pytest.fixture()
    def fixture():
        import eradiate
        eradiate.set_mode(mode)

    globals()['mode_' + mode] = fixture


for mode in eradiate._registered_modes:
    generate_fixture(mode)
del generate_fixture


# Add slow test marker
def pytest_configure(config):
    markexpr = config.getoption("markexpr", 'False')
    if not 'not slow' in markexpr:
        print("""\033[93mRunning the full test suite. To skip slow tests, please run "pytest -m 'not slow'"\033[0m""")

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
