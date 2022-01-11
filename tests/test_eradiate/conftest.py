import pytest

import eradiate

# ------------------------------------------------------------------------------
#                                 Mode fixtures
# ------------------------------------------------------------------------------


def generate_fixture(mode):
    @pytest.fixture()
    def fixture():
        import eradiate

        eradiate.set_mode(mode)

    globals()["mode_" + mode] = fixture


for mode in eradiate.modes():
    generate_fixture(mode)
del generate_fixture


def generate_fixture_group(name, modes):
    @pytest.fixture(params=modes)
    def fixture(request):
        mode = request.param
        import eradiate

        eradiate.set_mode(mode)

    globals()["modes_" + name] = fixture


variant_groups = {
    "all_mono": [x for x in eradiate.modes() if x.startswith("mono")],
    "all_ckd": [x for x in eradiate.modes() if x.startswith("ckd")],
    "all_mono_ckd": [
        x for x in eradiate.modes() if (x.startswith("mono") or x.startswith("ckd"))
    ],
    "all_single": [x for x in eradiate.modes() if not x.endswith("double")],
    "all_double": [x for x in eradiate.modes() if x.endswith("double")],
    "all": list(eradiate.modes()),
}

for name, variants in variant_groups.items():
    generate_fixture_group(name, variants)
del generate_fixture_group


# ------------------------------------------------------------------------------
#                              Other configuration
# ------------------------------------------------------------------------------


@pytest.fixture
def ert_seed_state():
    from eradiate.rng import SeedState

    return SeedState(0)
