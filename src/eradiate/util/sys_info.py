"""
System information display. This script is useful to collect environment
information. It is the fundamental building block of the `eradiate sys-info`
command.

This code is partly taken from `mitsuba.sys_info`.
"""

import importlib
import locale
import platform
import re
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def _run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if sys.platform.startswith("win32"):
        enc = "oem"
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def _run_and_match(command, regex):
    """Runs command and returns the first regex match if it exists"""
    rc, out, _ = _run(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def _get_cpu_info():
    if sys.platform.startswith("win32"):
        return platform.processor()
    elif sys.platform.startswith("darwin"):
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        ).strip()
    elif sys.platform.startswith("linux"):
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(r".*model name.*:", "", line, 1)[1:]
    return None


def show() -> dict:
    import drjit as dr
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    result = {}

    if sys.platform.startswith("darwin"):
        result["os"] = _run_and_match("sw_vers -productVersion", r"(.*)")
    elif sys.platform.startswith("win32"):
        result["os"] = platform.platform(terse=True)
    else:
        result["os"] = _run_and_match("lsb_release -a", r"Description:\t(.*)")

    result["cpu_info"] = _get_cpu_info()
    result["python"] = sys.version.replace("\n", "")
    result["llvm_version"] = dr.llvm_version()
    result["drjit_version"] = dr.__version__ + (" (DEBUG)" if dr.DEBUG else "")
    result["mitsuba_version"] = mi.MI_VERSION + (" (DEBUG)" if mi.DEBUG else "")
    try:
        eradiate_mitsuba_version = version("eradiate-mitsuba")
    except PackageNotFoundError:
        eradiate_mitsuba_version = f"{mi.ERD_MI_VERSION}, not installed [DEV|DEBUG]"
    result["eradiate_mitsuba_version"] = eradiate_mitsuba_version
    result["mitsuba_compiler"] = import_module("mitsuba.config").CXX_COMPILER

    # Python dependencies
    for package in ["numpy", "scipy", "xarray"]:
        try:
            mod = importlib.import_module(package)
            result[package] = mod.__version__
        except PackageNotFoundError:
            result[package] = "not found"

    return result


if __name__ == "__main__":
    sys_info = show()

    print("System")
    print(f"  CPU: {sys_info['cpu_info']}")
    print(f"  OS: {sys_info['os']}")
    print(f"  Python: {sys_info['python']}")

    print("\nVersions")
    print(f"  llvm {sys_info['llvm_version']}")
    print(f"  drjit {sys_info['drjit_version']}")
    print(
        f"  eradiate-mitsuba {sys_info['eradiate_mitsuba_version']} "
        f"(based on mitsuba {sys_info['mitsuba_version']})"
    )

    print("\nMitsuba variants")
    print("\n".join([f"  {variant}" for variant in mi.variants()]))
