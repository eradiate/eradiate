from collections import OrderedDict

from ._blind import BlindDataStore
from ._directory import DirectoryDataStore
from ._multi import MultiDataStore
from ._online import OnlineDataStore
from .._config import config

#: Global data store.
data_store = MultiDataStore(
    stores=OrderedDict(
        [
            (
                "small_files",
                DirectoryDataStore(path=config.dir / "resources/data/"),
            ),
            (
                "large_files_stable",
                OnlineDataStore(
                    base_url="http://eradiate.eu/data/store/stable/",
                    path=config.download_dir / "stable",
                ),
            ),
            (
                "large_files_unstable",
                BlindDataStore(
                    base_url="http://eradiate.eu/data/store/unstable/",
                    path=config.download_dir / "unstable",
                ),
            ),
        ]
    )
)
