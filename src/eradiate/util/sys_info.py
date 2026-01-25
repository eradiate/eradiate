"""
System information collection and display.

This module provides utilities to collect environment information including
system details, CPU, RAM, GPU, and package versions. It is the fundamental
building block of the `eradiate sys-info` command and pytest metrics.

This code is originally taken from `mitsuba.sys_info`.
"""

from __future__ import annotations

import locale
import os
import platform
import re
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import attrs

# -----------------------------------------------------------------------------
#                              Mixins and Base Classes
# -----------------------------------------------------------------------------


class DictMixin:
    """Mixin providing standard to_dict/from_dict for attrs classes."""

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


# -----------------------------------------------------------------------------
#                              Helper Functions
# -----------------------------------------------------------------------------


def _run(command: str) -> tuple[int, str, str]:
    """
    Run a shell command and return (return-code, stdout, stderr).

    Parameters
    ----------
    command
        Shell command to execute.

    Returns
    -------
    tuple[int, str, str]
        Tuple of (return_code, stdout, stderr).
    """
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


def _run_and_match(command: str, regex: str) -> str | None:
    """
    Run command and return the first regex match if it exists.

    Parameters
    ----------
    command
        Shell command to execute.
    regex
        Regular expression pattern with one capture group.

    Returns
    -------
    str or None
        First captured group from regex match, or None if no match.
    """
    rc, out, _ = _run(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def _get_nvidia_smi() -> str:
    """Get the path to nvidia-smi executable."""
    smi = "nvidia-smi"
    if sys.platform.startswith("win32"):
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = os.path.join(
            program_files_root, "NVIDIA Corporation", "NVSMI", smi
        )
        new_path = os.path.join(system_root, "System32", smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = f'"{candidate_smi}"'
                break
    return smi


def _get_gpu_info() -> list[str] | None:
    """
    Get list of GPU names using nvidia-smi.

    Returns
    -------
    list[str] or None
        List of GPU names, or None if nvidia-smi is not available.
    """
    smi = _get_nvidia_smi()
    rc, out, _ = _run(smi + " -L")
    if rc != 0:
        return None

    # Parse output: each line is like "GPU 0: NVIDIA GeForce RTX 3080 (UUID: ...)"
    gpus = []
    for line in out.strip().split("\n"):
        if not line:
            continue
        # Remove "GPU N: " prefix and "(UUID: ...)" suffix
        line = re.sub(r"GPU \d+:\s*", "", line)
        line = re.sub(r"\s*\(UUID:.*\)", "", line)
        if line:
            gpus.append(line.strip())

    return gpus if gpus else None


def _get_cuda_version() -> str | None:
    """
    Get CUDA version from nvidia-smi.

    Returns
    -------
    str or None
        CUDA version string (e.g., "12.4"), or None if not available.
    """
    smi = _get_nvidia_smi()
    rc, out, _ = _run(smi)
    if rc != 0:
        return None

    # Parse "CUDA Version: X.Y" from nvidia-smi output
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    if match:
        return match.group(1)
    return None


def _get_cpu_info() -> str | None:
    """
    Get CPU model name.

    Returns
    -------
    str or None
        CPU model name, or None if not available.
    """
    if sys.platform.startswith("win32"):
        return platform.processor()
    elif sys.platform.startswith("darwin"):
        try:
            result = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
            )
            return result.decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    elif sys.platform.startswith("linux"):
        try:
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(r".*model name.*:", "", line, 1).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return None


def _get_ram_info() -> float | None:
    """
    Get total RAM in GB.

    Returns
    -------
    float or None
        Total RAM in GB, or None if not available.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in kB
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)  # Convert to GB
        except (FileNotFoundError, ValueError, IndexError):
            return None
    elif sys.platform.startswith("darwin"):
        try:
            result = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])
            bytes_ram = int(result.decode().strip())
            return bytes_ram / (1024**3)  # Convert to GB
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None
    elif sys.platform.startswith("win32"):
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                return mem_status.ullTotalPhys / (1024**3)
        except Exception:
            return None
    return None


def _get_os_info() -> str | None:
    """
    Get detailed OS information.

    Returns
    -------
    str or None
        OS description string, or None if not available.
    """
    if sys.platform.startswith("darwin"):
        return _run_and_match("sw_vers -productVersion", r"(.*)")
    elif sys.platform.startswith("win32"):
        return platform.platform(terse=True)
    else:
        return _run_and_match("lsb_release -a", r"Description:\t(.*)")


# -----------------------------------------------------------------------------
#                              Data Classes
# -----------------------------------------------------------------------------


@attrs.define
class GitInfo(DictMixin):
    """Git repository information."""

    commit_hash: str | None = None
    commit_time: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    @classmethod
    def collect(cls) -> GitInfo:
        """
        Collect git information from the current repository.

        Returns
        -------
        GitInfo
            Git information with commit hash, time, branch, and dirty status.
        """
        info = cls()

        try:
            result = _run("git rev-parse HEAD")
            info.commit_hash = result[1].strip()

            result = _run("git log -1 --format=%cI")
            info.commit_time = result[1].strip()

            result = _run("git rev-parse --abbrev-ref HEAD")
            info.branch = result[1].strip()

            result = _run("git status --porcelain")
            info.dirty = bool(result[1].strip())

        except Exception:
            pass

        return info


@attrs.define
class SysInfo(DictMixin):
    """
    System information container.

    Collects and stores system details including hostname, platform, CPU,
    RAM, GPU, Python version, package versions, and Mitsuba/Dr.Jit details.
    """

    # Basic system info
    hostname: str | None = None
    platform: str | None = None
    platform_release: str | None = None
    os: str | None = None
    cpu: str | None = None
    ram_gb: float | None = None
    llvm_version: str | None = None
    gpus: list[str] | None = None
    cuda_version: str | None = None
    python: str | None = None

    # Package versions
    packages: dict[str, str] = attrs.field(factory=dict)

    # Mitsuba/Dr.Jit specific
    drjit_version: str | None = None
    mitsuba_version: str | None = None
    eradiate_mitsuba_version: str | None = None
    mitsuba_compiler: str | None = None

    @classmethod
    def collect(cls, packages: list[str] | None = None) -> SysInfo:
        """
        Collect system information.

        Parameters
        ----------
        packages
            List of package names to include version info for. Defaults to
            common scientific packages used by Eradiate.

        Returns
        -------
        SysInfo
            System information with all collected fields populated.
        """
        import drjit as dr
        import mitsuba as mi

        mi.set_variant("scalar_rgb")

        if packages is None:
            packages = [
                "eradiate",
                "numpy",
                "scipy",
                "xarray",
            ]

        pkg_versions = {}
        for pkg_name in packages:
            try:
                pkg_versions[pkg_name] = version(pkg_name)
            except PackageNotFoundError:
                pass

        # Get eradiate-mitsuba version
        try:
            eradiate_mitsuba_version = version("eradiate-mitsuba")
        except PackageNotFoundError:
            eradiate_mitsuba_version = f"{mi.ERD_MI_VERSION}, not installed [DEV]"

        return cls(
            hostname=platform.node(),
            platform=sys.platform,
            platform_release=platform.release(),
            os=_get_os_info(),
            cpu=_get_cpu_info(),
            ram_gb=_get_ram_info(),
            gpus=_get_gpu_info(),
            cuda_version=_get_cuda_version(),
            python=sys.version.replace("\n", ""),
            packages=pkg_versions,
            llvm_version=",".join(list(map(str, dr.detail.llvm_version()))),
            drjit_version=dr.__version__,
            mitsuba_version=mi.MI_VERSION,
            eradiate_mitsuba_version=eradiate_mitsuba_version,
            mitsuba_compiler=import_module("mitsuba.config").CXX_COMPILER,
        )


if __name__ == "__main__":
    # Use this for quick checks of the collection process
    from rich.pretty import pprint

    pprint(SysInfo.collect())
    pprint(GitInfo.collect())
