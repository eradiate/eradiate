from __future__ import annotations

from collections import OrderedDict

from ._blind_directory import BlindDirectoryDataStore
from ._blind_online import BlindOnlineDataStore
from ._multi import MultiDataStore
from ._safe_directory import SafeDirectoryDataStore
from ._safe_online import SafeOnlineDataStore
from .. import config

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

    download_dir = config.settings.download_dir
    source_dir = config.SOURCE_DIR
    data_dir = None if source_dir is None else config.SOURCE_DIR / "resources" / "data"

    if offline is None:
        offline = config.settings.offline

    if production is None:
        production = config.SOURCE_DIR is None

    # 'small_files' configuration
    if offline:
        if production:
            # Offline, prod: we expect files to be in the download directory
            small_files = BlindDirectoryDataStore(path=download_dir)
        else:
            # Offline, dev: we expect files to be in the data submodule
            small_files = SafeDirectoryDataStore(path=data_dir)
    else:
        if production:
            # Online, prod: we expect files to be on GitHub
            small_files = SafeOnlineDataStore(
                base_url="/".join(
                    [
                        config.settings.small_files_registry_url,
                        config.settings.small_files_registry_revision,
                    ]
                ),
                path=download_dir,
            )
        else:
            # Online, dev: we expect files to be in the data submodule
            small_files = SafeDirectoryDataStore(path=data_dir)

    # 'large_files_stable' configuration
    if offline:
        # Offline: we expect files to be in the download directory
        large_files_stable = BlindDirectoryDataStore(path=download_dir / "stable")
    else:
        # Online: we expect files to be on the Internet, with checksum
        large_files_stable = SafeOnlineDataStore(
            base_url="/".join([config.settings.data_store_url, "stable"]),
            path=download_dir / "stable",
        )

    # 'large_files_unstable' configuration
    if offline:
        # Offline: we expect files to be in the download directory
        large_files_unstable = BlindDirectoryDataStore(path=download_dir / "unstable")
    else:
        # Online: we expect files to be on the Internet, without checksum
        large_files_unstable = BlindOnlineDataStore(
            base_url="/".join([config.settings.data_store_url, "unstable"]),
            path=download_dir / "unstable",
        )

    data_store = MultiDataStore(
        stores=OrderedDict(
            [
                ("small_files", small_files),
                ("large_files_stable", large_files_stable),
                ("large_files_unstable", large_files_unstable),
            ]
        )
    )


# Initialise the data store upon module import
init_data_store()
