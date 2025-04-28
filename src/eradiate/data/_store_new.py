from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from pathlib import Path

import attrs

from .._version import _version
from ..attrs import define, documented
from ..config import settings

logger = logging.getLogger(__file__)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@define
class AssetManager:
    """
    A lightweight asset manager for Eradiate data.

    Examples
    --------
    >>> from eradiate.data import asset_manager
    >>> asset_manager.install(["core", "komodo", "monotropa"])
    """

    cache_dir: Path = documented(
        attrs.field(converter=lambda x: Path(x).expanduser().resolve()),
        doc="Cache location on disk.",
        type="Path",
        init_type="path-like",
    )

    unpack_dir: Path = documented(
        attrs.field(converter=lambda x: Path(x).expanduser().resolve()),
        doc="Resource unpacking location on disk.",
        type="Path",
    )

    install_dir: Path = documented(
        attrs.field(converter=lambda x: Path(x).expanduser().resolve()),
        doc="Install location on disk.",
        type="Path",
        init_type="path-like",
    )

    base_url: str = documented(
        attrs.field(converter=lambda x: x.rstrip("/")),
        doc="Base URL of remote storage.",
        type="str",
    )

    offline: bool = documented(attrs.field(default=False))

    _resources: dict = attrs.field(factory=dict, repr=False)

    _manifest: dict | None = attrs.field(default=None, repr=False)

    def __attrs_post_init__(self):
        for path in [
            self.cache_dir,
            self.unpack_dir,
            self.install_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self._manifest = self.read_manifest()
        self._resources = self.read_resources()

    def update_manifest(self) -> None:
        """
        Update cached manifest file from the remote resource store.
        """
        raise NotImplementedError

    def read_manifest(self) -> dict:
        """
        Read manifest from disk.
        """
        raise NotImplementedError

    def read_resources(self) -> dict:
        """
        Read resource alias list from disk.
        """
        raise NotImplementedError

    def resolve_resource(self, resource_id: str) -> list[str]:
        """
        Resolve resource alias.
        """
        raise NotImplementedError

    def list(self) -> None:
        """
        List available resources and their state.
        """
        raise NotImplementedError

    def download(self, resource_id: str) -> None:
        """
        Download a single resource from remote storage.
        """
        raise NotImplementedError

    def unpack(self, resource_id: str):
        """
        Unpack a single resource from local storage.
        """
        raise NotImplementedError

    def install(self, resource_id: str):
        """
        Install a single resource and make it available for consumption.
        """
        raise NotImplementedError

    def clear_cache(self) -> None:
        raise NotImplementedError

    def walk(self, **kwargs):
        """
        Walk the installation directory tree. This function is a convenience
        wrapper around :func:`os.walk`.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :func:`os.walk`.
        """
        yield from os.walk(str(self.assets_dir), **kwargs)

    def list_files(self, pattern: str = "*") -> list[Path]:
        result = []

        # Walk through directory structure
        for current_depth, (root, dirs, files) in enumerate(self.walk()):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    result.append(Path(root) / filename)

        return result


#: Unique asset manager instance
asset_manager = AssetManager(
    cache_dir=Path(settings["data_path"]) / "resources",
    unpack_dir=Path(settings["data_path"]) / "unpacked",
    install_dir=Path(settings["data_path"]) / "assets" / f"eradiate-v{_version}",
    offline=settings["offline"],
)
