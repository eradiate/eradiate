"""
Eradiate's computational kernel, based on the Mitsuba 2 rendering system.
"""

try:
    import mitsuba
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'mitsuba'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e

try:
    getattr(mitsuba, "ERADIATE_KERNEL")
except AttributeError as e:
    raise RuntimeError(
        "The imported 'mitsuba' module is not the Eradiate kernel version."
    ) from e
