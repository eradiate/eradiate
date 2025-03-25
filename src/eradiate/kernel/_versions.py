"""
Upon import, this module checks if the kernel dependencies are correctly set up.
"""

from importlib.metadata import PackageNotFoundError, version

# Internal constants
REQUIRED_MITSUBA_VERSION = "3.5.2"
REQUIRED_MITSUBA_PATCH_VERSION = "0.3.2"


def find_drjit():
    # Check if Dr.Jit and Mitsuba can be imported successfully
    try:
        __import__("drjit")
        return True

    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import module 'drjit'. If you are an end-user, you need "
            "to install the radiometric kernel by adding the 'kernel' extra to "
            "the installation command (e.g. pip install 'eradiate[kernel]'). "
            "If you are a developer, you have to build the kernel and add "
            "it to your $PYTHONPATH."
        ) from e


def find_mitsuba():
    # Check if Dr.Jit and Mitsuba can be imported successfully
    try:
        __import__("mitsuba")
        return True

    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import module 'mitsuba'. If you are an end-user, you need "
            "to install the radiometric kernel by adding the 'kernel' extra to "
            "the installation command (e.g. pip install 'eradiate[kernel]'). "
            "If you are a developer, you have to build the kernel and add "
            "it to your $PYTHONPATH."
        ) from e


def kernel_version():
    """
    Detect the version of the Eradiate kernel.

    Returns
    -------
    mitsuba_version : str
        The version of Mitsuba on which the kernel is based. This is the
        :data:`mitsuba.MI_VERSION` constant.

    patch_version : str or None
        The version of the kernel. This is the :data:`mitsuba.ERD_MI_VERSION`
        constant. If the constant is not found, *i.e.* if the detected Mitsuba
        package is not the Eradiate kernel, ``None`` is returned.

    Notes
    -----
    This function reads the metadata exposed by the ``mitsuba`` module, not
    package metadata.
    """
    # Retrieve the kernel and kernel patch versions
    _mi = __import__("mitsuba")
    mitsuba_version = _mi.scalar_rgb.MI_VERSION
    patch_version = (
        _mi.scalar_rgb.ERD_MI_VERSION
        if hasattr(_mi.scalar_rgb, "ERD_MI_VERSION")
        else None
    )
    return mitsuba_version, patch_version


def kernel_installed():
    """
    Check if the kernel Python package is installed to the active Python
    environment.

    Returns
    -------
    is_installed : bool
        ``True`` iff ``eradiate-mitsuba`` package is installed to the active
        Python environment.

    package_version : str or None
         If ``is_installed`` is ``True``, a string containing the version found
         for that package.
    """

    # Check if the eradiate-mitsuba package is installed
    try:
        return True, version("eradiate-mitsuba")
    except PackageNotFoundError:
        return False, None


def check_kernel():
    """
    Perform a few basic checks on the kernel.
    """
    warnings = []

    # Can we find Dr.Jit in the current Python path?
    find_drjit()

    # Can we find Mitsuba in the current Python path?
    find_mitsuba()

    # Is the kernel version compatible with requirements?
    mitsuba_version, patch_version = kernel_version()

    if patch_version is None:
        warnings.append(
            "Detected Mitsuba build is not the Eradiate Mitsuba version. "
            "Make sure to use the Eradiate-specific version of Mitsuba, compiled "
            "with the appropriate variants and plugins."
        )
    elif (
        patch_version != REQUIRED_MITSUBA_PATCH_VERSION
        or mitsuba_version != REQUIRED_MITSUBA_VERSION
    ):
        warnings.append(
            f"Found incompatible kernel version "
            f"{patch_version}, based on Mitsuba {mitsuba_version}. "
            f"Required: {REQUIRED_MITSUBA_PATCH_VERSION}, based on Mitsuba "
            f"{REQUIRED_MITSUBA_VERSION}."
        )

    return warnings


check_kernel()
