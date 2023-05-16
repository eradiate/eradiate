import pytest

import eradiate

# ------------------------------------------------------------------------------
#                            Pre-process helpers
# ------------------------------------------------------------------------------

pytest.register_assert_rewrite("eradiate.test_tools.types.check_scene_element")

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


variants = [x for x in eradiate.modes() if x not in {"mono", "ckd"}]  # Remove aliases
variant_groups = {
    "all_mono": [x for x in variants if x.startswith("mono")],
    "all_ckd": [x for x in variants if x.startswith("ckd")],
    "all_mono_ckd": [
        x for x in variants if (x.startswith("mono") or x.startswith("ckd"))
    ],
    "all_single": [x for x in variants if x.endswith("single")],
    "all_double": [x for x in variants if x.endswith("double")],
    "all": variants,
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
