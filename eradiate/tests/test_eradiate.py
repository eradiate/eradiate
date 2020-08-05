"""Root module testing."""


def test_modes():
    import eradiate

    # Check that switching to mono mode works
    eradiate.set_mode("mono")
    # We expect that the kernel variant is appropriately selected
    assert eradiate.kernel.variant() == "scalar_mono_double"
    # We check for defaults
    assert eradiate.mode.config["wavelength"] == 550.

    # Check that defaults are correctly applied
    eradiate.set_mode("mono", wavelength=300.)
    assert eradiate.mode.config["wavelength"] == 300.
