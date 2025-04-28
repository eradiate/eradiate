from __future__ import annotations

import enum
import importlib.resources
import json
import logging
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path, PosixPath
from typing import ClassVar, Literal
from urllib.parse import urljoin

import attrs
import pooch
from ruamel.yaml import YAML

from .._version import _version
from ..attrs import define, documented
from ..config import settings
from ..units import unit_registry as ureg

logger = logging.getLogger(__file__)
_yaml = YAML()


class ResourceState(enum.Flag):
    NONE = enum.auto()
    CACHED = enum.auto()
    UNPACKED = enum.auto()
    INSTALLED = enum.auto()

    @staticmethod
    def to_string(value) -> str:
        result = [
            "C" if value & ResourceState.CACHED else "-",
            "U" if value & ResourceState.UNPACKED else "-",
            "I" if value & ResourceState.INSTALLED else "-",
        ]
        return "".join(result)


@define
class Resource:
    keyword: str
    hash: str
    type: str
    size: int

    #: Maps archive types to corresponding file extensions
    _EXTENSIONS: ClassVar[dict] = {"zip": "zip"}

    def filename(self) -> str:
        return f"{PosixPath(self.keyword)}.{Resource._EXTENSIONS[self.type]}"


@define
class AssetManager:
    """
    A lightweight asset manager for Eradiate data.

    Examples
    --------
    >>> from eradiate.data import asset_manager
    >>> asset_manager.update()
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
        attrs.field(converter=str),
        doc="Base URL of remote storage.",
        type="str",
    )

    #: Manifest file contents, as a dictionary.
    _manifest: dict | None = attrs.field(default=None, repr=False)

    #: Resource alias list, as a dictionary.
    _resource_aliases: dict = attrs.field(factory=dict, repr=False)

    #: List of installed resources, as a set.
    _installed: set = attrs.field(factory=set, repr=False)

    @property
    def manifest_path(self) -> Path:
        """
        Path to the manifest file.
        """
        return self.cache_dir / "manifest.json"

    @property
    def installed_path(self) -> Path:
        """
        Path to the list of installed resources.
        """
        return self.install_dir / "installed.json"

    def __attrs_post_init__(self):
        # Make sure that paths point to defined directories
        for path in [self.cache_dir, self.unpack_dir, self.install_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Load metadata files
        self._manifest = self._load_manifest()
        self._resource_aliases = self._load_resource_aliases()
        self._installed = self._load_installed()

    def _update_manifest(self) -> None:
        """
        Update cached manifest file from the remote resource store.
        """
        url = urljoin(self.base_url, "manifest.json")
        print(f"Downloading manifest from {url}")
        urllib.request.urlretrieve(url, self.manifest_path)

    def _load_manifest(self) -> dict | None:
        """
        Read and parse manifest from disk.
        """
        manifest_path = self.manifest_path

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            return {
                resource_id: Resource(**manifest["resources"][resource_id])
                for resource_id in manifest["resources"]
            }
        else:
            return None

    def _load_installed(self) -> set:
        """
        Read and parse list of installed resources from disk.
        """
        installed_path = self.installed_path

        if installed_path.exists():
            with open(installed_path) as f:
                installed = json.load(f)
            return set(installed)
        else:
            return set()

    @staticmethod
    def _load_resource_aliases() -> dict:
        """
        Read resource alias mapping from disk.
        """
        return _yaml.load(
            Path(importlib.resources.files("eradiate.data") / "resources.yml")
        )["resources"]

    def _resolve_alias(self, resource_ids: str | list[str]) -> list[str]:
        """
        Resolve one or several resource aliases. Aliases are recursively
        resolved until the referenced resource is no longer an alias (and
        presumably points to a resource that can be found in the manifest).

        Parameters
        ----------
        resource_ids : str or list of str
            One or several resource IDs or aliases.

        Returns
        -------
        list of str
            A list of resolved resource IDs.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        result = []

        for resource_id in resource_ids:
            if resource_id not in self._resource_aliases:
                result.append(resource_id)
            else:
                result.extend(self._resolve_alias(self._resource_aliases[resource_id]))

        return result

    def _resolve_resource(self, resource_ids: str | list[str]) -> list[Resource]:
        """
        Return a resource definition given its ID.
        """
        resource_ids = self._resolve_alias(resource_ids)
        result = []

        for resource_id in resource_ids:
            result.append(self._manifest[resource_id])

        return result

    def _get_path_remote(self, resource_id: str) -> str:
        """
        Get URL to a given resource file on the remote storage.
        """
        resource = self._manifest[resource_id]
        fname = resource.filename()
        return urljoin(self.base_url, str(fname))

    def _get_path_cache(self, resource_id: str) -> Path:
        """
        Get cache path for a given resource.
        """
        resource = self._manifest[resource_id]
        fname = resource.filename()
        return self.cache_dir / Path(fname)

    def _get_path_unpack(self, resource_id: str) -> Path:
        """
        Get unpack path for a given resource.
        """
        resource = self._manifest[resource_id]
        fname = resource.filename()
        return Path(self.unpack_dir / Path(fname)).with_suffix("")

    def _list_files(self, resource_id: str) -> list[Path]:
        """
        List unpacked files for a given resource.
        """
        resource_dir = self._get_path_unpack(resource_id)
        result = []

        for root, dirs, files in os.walk(str(resource_dir)):
            for filename in files:
                result.append(Path(root) / filename)

        return result

    def _install_remove(
        self, resource_id: str, mode: Literal["install", "remove"]
    ) -> None:
        """
        Install or remove a single resource.
        """
        file_list = self._list_files(resource_id)
        unpack_root = self._get_path_unpack(resource_id)

        if mode == "install":
            print(f"Installing resource '{resource_id}'")
            for src in file_list:
                dst = self.install_dir / src.relative_to(unpack_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src)
            self._installed.add(resource_id)

        elif mode == "remove":
            print(f"Removing resource '{resource_id}'")
            for src in file_list:
                dst = self.install_dir / src.relative_to(unpack_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
            if resource_id in self._installed:
                self._installed.remove(resource_id)

        else:
            raise ValueError(f"unknown mode '{mode}'")

        with open(self.installed_path, "w") as f:
            json.dump(list(self._installed), f)

    def _state(self, resource_ids: str | list[str]) -> list[ResourceState]:
        """
        Return information about the state of a given resource.

        Each resource has the following possible states:

        * cached: the archive is downloaded and found in the cache directory;
        * unpacked: the archive is extracted to the unpack location;
        * installed: the resource is installed to the installation directory.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        result = []

        for resource_id in resource_ids:
            state = ResourceState.NONE

            path_cache = self._get_path_cache(resource_id)
            if path_cache.exists():
                state |= ResourceState.CACHED

            path_unpack = self._get_path_unpack(resource_id)
            if path_unpack.exists():
                state |= ResourceState.UNPACKED

            if resource_id in self._installed:
                state |= ResourceState.INSTALLED

            result.append(state)

        return result

    # --------------------------------------------------------------------------
    #                              Public interface
    # --------------------------------------------------------------------------

    def update(self, download: bool | None = None) -> None:
        """
        Update the manifest, either reading it from disk if offline or recent
        enough, or downloading it from remote storage.

        Parameters
        ----------
        download : bool, optional
            If ``True``, download the manifest from remote storage first.
            If ``False``, only read from disk.
            If ``None`` (unset), apply default policy (download if the file on
            disk is more than a day old and not in offline mode).
        """
        if download is None:
            if self.manifest_path.exists():
                manifest_age = time.time() - os.path.getmtime(self.manifest_path)
            else:
                manifest_age = float("inf")
            download = not settings.get("OFFLINE") and (manifest_age > 86400.0)

        if download:
            self._update_manifest()

        self._manifest = self._load_manifest()

    def info(self):
        """
        Print information about the asset manager to the terminal.
        """
        print("Eradiate asset manager")
        print(f"* Remote storage URL: {self.base_url}")
        print(f"* Asset cache location: {self.cache_dir}")
        print(f"* Unpacked asset location: {self.unpack_dir}")
        print(f"* Installation location: {self.install_dir}")

    def list(
        self,
        what: Literal["resources", "aliases", "all"]
        | list[Literal["resources", "aliases"]] = "resources",
    ) -> None:
        """
        Display the list available resources in the manifest and their state.

        Parameters
        ----------
        what : {"resources", "aliases", "all"}
            One or several keywords that specify what to show.
        """
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if what == "all":
            what = ["resources", "aliases"]

        if "resources" in what:
            # List available resources and their state
            manifest = {} if self._manifest is None else self._manifest
            available_resources = []
            for resource_id, resource in manifest.items():
                kw = resource.keyword
                size = (resource.size * ureg("B")).to_compact()
                type = resource.type
                state = ResourceState.to_string(self._state(resource_id)[0])
                available_resources.append((kw, type, size, state))

            available_resources.sort(key=lambda x: x[0])

            table_resources = Table(box=box.SIMPLE)
            table_resources.add_column("ID")
            table_resources.add_column("Type")
            table_resources.add_column("Size")
            table_resources.add_column("State")

            for x in available_resources:
                table_resources.add_row(x[0], x[1], f"{x[2]:.3g~P}", x[3])
            console.print(table_resources)

        if "aliases" in what:
            # List and render resource aliases
            aliases = self._resource_aliases
            table_aliases = Table(box=box.SIMPLE)
            table_aliases.add_column("Alias")
            table_aliases.add_column("Target")

            for k, v in aliases.items():
                if not isinstance(v, (list, tuple)):
                    v = [v]

                for i, x in enumerate(v):
                    if i == 0:
                        table_aliases.add_row(k, x)
                    else:
                        table_aliases.add_row("", x)
            console.print(table_aliases)

    def download(
        self,
        resource_ids: str | list[str],
        unpack: bool = True,
        progressbar: bool = True,
    ) -> None:
        """
        Download a resource from remote storage to the cache directory.

        Parameters
        ----------
        resource_ids : str or list of str
            Resource ID(s) to download.

        unpack : bool, default: True
            Unpack downloaded archive.

        progressbar : bool, default: True
            If ``True``, display a progress bar for the download task.

        Notes
        -----
        This function uses the :func:`pooch.retrieve` function under the hood.
        """

        resources = self._resolve_resource(resource_ids)

        for resource in resources:
            fname = resource.filename()
            url = urljoin(self.base_url, fname)

            if unpack:
                if resource.type == "zip":
                    processor = pooch.processors.Unzip(
                        extract_dir=self.unpack_dir
                        / Path(resource.filename()).with_suffix("")
                    )
                else:
                    raise ValueError(
                        f"cannot unpack unhandled resource type '{resource.type}'"
                    )
            else:
                processor = None

            pooch.retrieve(
                url,
                fname=fname,
                path=self.cache_dir,
                known_hash=f"md5:{resource.hash}",
                processor=processor,
                progressbar=progressbar,
            )

    def install(self, resource_ids: str | list[str], progressbar: bool = True) -> None:
        """
        Install a single resource and make it available for consumption.

        Parameters
        ----------
        resource_ids : str or list of str
            Resource ID(s) to install. The system ensures that requested
            resources are downloaded and unpacked.

        progressbar : bool, default: True
            If ``True``, display a progress bar if a download task is triggered.
        """
        self.update()
        self.download(resource_ids, unpack=True, progressbar=progressbar)
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        resource_ids = self._resolve_alias(resource_ids)

        for resource_id in resource_ids:
            self._install_remove(resource_id, mode="install")

    def remove(self, resource_ids: str | list[str]) -> None:
        """
        Uninstall a single resource.

        Parameters
        ----------
        resource_ids : str or list of str
            Resource ID(s) to uninstall.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        resource_ids = self._resolve_alias(resource_ids)

        for resource_id in resource_ids:
            self._install_remove(resource_id, mode="remove")

    def clear(
        self,
        resource_ids: str | list[str] | None = None,
        what: Literal["cache", "unpack", "install", "all"]
        | list[Literal["cache", "unpack", "install"]] = "cache",
    ) -> None:
        """
        Clear the cache directory and save up disk space.

        Parameters
        ----------
        resource_ids : str or list of str, optional
            Resource(s) for which to clear data. If unset, all data will be
            wiped.

        what : {"cache", "unpack", "install", "all"}, optional
            One or several keywords that specify which part of the data to
            clear.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]

        if what == "all":
            what = ["cache", "unpack", "install"]

        if "install" in what:
            if resource_ids is not None:
                self.remove(resource_ids)
            else:
                print("Clearing all installed resources")
                for path in self.install_dir.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                self._installed.clear()
                with open(self.installed_path, "w") as f:
                    json.dump(list(self._installed), f)

        if "unpack" in what:
            if resource_ids is not None:
                for resource_id in resource_ids:
                    print("Removing unpacked resource", resource_id)
                    path = self._get_path_unpack(resource_id)
                    shutil.rmtree(path, ignore_errors=True)
            else:
                print("Clearing all unpacked resources")
                for fname in self.unpack_dir.iterdir():
                    if fname.is_dir():
                        shutil.rmtree(fname, ignore_errors=True)

        if "cache" in what:
            if resource_ids is not None:
                for resource_id in resource_ids:
                    print("Removing cached resource", resource_id)
                    path = self._get_path_cache(resource_id)
                    shutil.rmtree(path, ignore_errors=True)
            else:
                print("Clearing all cache")
                for fname in self.cache_dir.iterdir():
                    if fname.is_dir() and fname != "manifest.json":
                        shutil.rmtree(fname, ignore_errors=True)


#: Unique asset manager instance
asset_manager = AssetManager(
    cache_dir=Path(settings["data_path"]) / "cached",
    unpack_dir=Path(settings["data_path"]) / "unpacked",
    install_dir=Path(settings["data_path"]) / "installed" / f"eradiate-v{_version}",
    base_url=settings["data_url"],
)
