from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from ._blind_directory import BlindDirectoryDataStore
from ._blind_online import BlindOnlineDataStore
from ._multi import MultiDataStore
from ._safe_directory import SafeDirectoryDataStore
from ._safe_online import SafeOnlineDataStore
from .._config import config

#: Global data store.
data_store: MultiDataStore | None = None


def init_data_store(
    offline: bool | None = None, production: bool | None = None
) -> None:
    """
    Initialize the global data store.

    Parameters
    ----------
    offline : bool, optional
        If ``True``, replace all online data stores with blind directory data
        stores. If unset, the global offline configuration is used.

    production : bool, optional
        If ``True``, replace all development data stores providing files from
        the current local source folder with online safe data stores. If set to
        ``True`` with ``offline`` flag set to ``True``, then the data must be
        provided by the user before accessing it. If unset, the global
        production configuration is used.

    Notes
    -----
    This function is called automatically when the ``eradiate.data`` package is
    imported.
    """
    global data_store

    if offline is None:
        offline = config.offline

    if production is None:
        production = config.source_dir is None

    download_dir = config.download_dir
    if download_dir is None:
        if production:
            download_dir = Path("./eradiate_downloads").absolute()
        else:
            download_dir = config.source_dir / "resources" / "data"

    if not production:
        small_files = SafeDirectoryDataStore(path=download_dir)
    else:
        if offline:
            small_files = BlindDirectoryDataStore(path=download_dir)
        else:
            small_files = SafeOnlineDataStore(
                base_url="/".join(
                    [
                        config.small_files_registry_url,
                        config.small_files_registry_revision,
                    ]
                ),
                path=download_dir,
            )

    if offline:
        data_store = MultiDataStore(
            stores=OrderedDict(
                [
                    ("small_files", small_files),
                    (
                        "large_files_stable",
                        BlindDirectoryDataStore(path=download_dir / "stable"),
                    ),
                    (
                        "large_files_unstable",
                        BlindDirectoryDataStore(path=download_dir / "unstable"),
                    ),
                ]
            )
        )

    else:
        data_store = MultiDataStore(
            stores=OrderedDict(
                [
                    ("small_files", small_files),
                    (
                        "large_files_stable",
                        SafeOnlineDataStore(
                            base_url="/".join([config.data_store_url, "stable"]),
                            path=download_dir / "stable",
                        ),
                    ),
                    (
                        "large_files_unstable",
                        BlindOnlineDataStore(
                            base_url="/".join([config.data_store_url, "unstable"]),
                            path=download_dir / "unstable",
                        ),
                    ),
                ]
            )
        )


# Initialise the data store upon module import
init_data_store()
