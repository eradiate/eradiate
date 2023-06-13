(sec-reference_cli)=
# Command-line interface reference

This is the reference for Eradiate’s command-line tools. It consists of a main entry point `eradiate`, and its multiple subcommands documented hereafter. The implementation is located in the `eradiate.cli` module (not documented here).

## `eradiate`

Eradiate — A modern radiative transfer model for Earth observation.

**Usage**:

```console
$ eradiate [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--log-level [CRITICAL|ERROR|WARNING|INFO|DEBUG|NOTSET]`: Set log level.  [default: LogLevel.WARNING]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `data`: Manage data.
* `show`: Display information useful for debugging.
* `srf`: Spectral response function filtering utility.

### `eradiate data`

Manage data.

**Usage**:

```console
$ eradiate data [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `fetch`: Fetch files from the Eradiate data store.
* `info`: Display information about data store...
* `make-registry`: Recursively construct a file registry from...
* `purge-cache`: Purge the cache of online data stores.
* `update-registries`: Update local registries for online sources.

#### `eradiate data fetch`

Fetch files from the Eradiate data store.

**Usage**:

```console
$ eradiate data fetch [OPTIONS] [FILE_LIST]...
```

**Arguments**:

* `[FILE_LIST]...`: An arbitrary number of relative paths to files to be retrieved from the data store. If unset, the list of files is read from a YAML file which can be specified by using the ``--from-file`` option and defaults to ``$ERADIATE_SOURCE_DIR/data/downloads.yml`` a production environment and ``$ERADIATE_SOURCE_DIR/data/downloads_development.yml`` in a development environment.

**Options**:

* `-f, --from-file TEXT`: Optional path to a file list (YAML format). If this option is set, the FILES argument(s) will be ignored.
* `--help`: Show this message and exit.

#### `eradiate data info`

Display information about data store configuration.

**Usage**:

```console
$ eradiate data info [OPTIONS] [DATA_STORES]...
```

**Arguments**:

* `[DATA_STORES]...`: List of data stores for which information is requested. If no data store ID is passed, information is displayed for all data stores.

**Options**:

* `-l, --list-registry`: Show registry content if relevant.
* `--help`: Show this message and exit.

#### `eradiate data make-registry`

Recursively construct a file registry from the current working directory.

**Usage**:

```console
$ eradiate data make-registry [OPTIONS]
```

**Options**:

* `-i, --input-directory TEXT`: Path to input directory.  [default: .]
* `-o, --output-file TEXT`: Path to output file (default: '<input_directory>/registry.txt').
* `-r, --rules TEXT`: Path to the registry rule file (default: '<input_directory>/registry_rules.yml').
* `-a, --hash-algorithm TEXT`: Hashing algorithm (default: 'sha256').  [default: sha256]
* `--help`: Show this message and exit.

#### `eradiate data purge-cache`

Purge the cache of online data stores.

**Usage**:

```console
$ eradiate data purge-cache [OPTIONS]
```

**Options**:

* `-k, --keep`: Keep registered files.
* `--help`: Show this message and exit.

#### `eradiate data update-registries`

Update local registries for online sources.

**Usage**:

```console
$ eradiate data update-registries [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `eradiate show`

Display information useful for debugging.

**Usage**:

```console
$ eradiate show [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `eradiate srf`

Spectral response function filtering utility.

**Usage**:

```console
$ eradiate srf [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `filter`: Filter a spectral response function data set.
* `trim`: Trim a spectral response function.

#### `eradiate srf filter`

Filter a spectral response function data set.

**Usage**:

```console
$ eradiate srf filter [OPTIONS] FILENAME OUTPUT
```

**Arguments**:

* `FILENAME`:  path to the spectral response function data to filter.  [required]
* `OUTPUT`: Path where to write the filtered data.  [required]

**Options**:

* `--trim / --no-trim`: Trim the data set prior to filtering.  [default: trim]
* `-v, --verbose`: Display filtering summary.
* `-d, --dry-run`: Do not write filtered data to file.
* `-i, --interactive`: Prompt user to proceed to saving the filtered dataset.
* `-s, --show-plot`: Show plot of the filtered region.
* `-t, --threshold FLOAT`: Data points where response is less then or equal to this value are dropped.
* `-w, --wmin FLOAT`: Lower wavelength value in nm.
* `-W, --wmax FLOAT`: Upper wavelength value in nm.
* `-p, --percentage FLOAT`: Data points that do not contribute to this percentage of the integrated spectral response are dropped
* `--help`: Show this message and exit.

#### `eradiate srf trim`

Trim a spectral response function.
Remove all-except-last leading zeros and all-except-first trailing zeros.

**Usage**:

```console
$ eradiate srf trim [OPTIONS] FILENAME OUTPUT
```

**Arguments**:

* `FILENAME`: [required]
* `OUTPUT`: File where to write the filtered data set.  [required]

**Options**:

* `-v, --verbose`: Display filtering summary.
* `-s, --show-plot`: Show plot of the filtered region.
* `-d, --dry-run`: Do not write filtered data to disk.
* `-i, --interactive`: Prompt user to proceed to saving the filtered dataset.
* `--help`: Show this message and exit.
