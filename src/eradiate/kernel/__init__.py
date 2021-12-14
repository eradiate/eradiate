"""
Eradiate's computational kernel is based on the Mitsuba 2 rendering system.
Eradiate imports the :mod:`mitsuba` module and makes a basic check to detect if
it is not our customised version.
"""

# Check if Mitsuba can be imported successfully
try:
    import mitsuba
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Could not import module 'mitsuba'; did you build the kernel and add "
        "it to your $PYTHONPATH?"
    ) from e


from . import gridvolume, logging, transform
from ._bitmap import bitmap_to_dataset
