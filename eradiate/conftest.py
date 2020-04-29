import pytest


def generate_fixture(variant):
    @pytest.fixture()
    def fixture():
        try:
            import eradiate.kernel
            eradiate.kernel.set_variant(variant)
        except Exception:
            pytest.skip('Mitsuba variant "%s" is not enabled!' % variant)
    globals()['variant_' + variant] = fixture


for variant in ['scalar_rgb', 'scalar_spectral',
                'scalar_mono', 'scalar_mono_polarized', 'packet_rgb',
                'packet_spectral']:
    generate_fixture(variant)
del generate_fixture
