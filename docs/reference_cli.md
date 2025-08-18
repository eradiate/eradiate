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

* `--log-level [CRITICAL|ERROR|WARNING|INFO|DEBUG|NOTSET]`: Set log level.  [default: WARNING]
* `--debug / --no-debug`: Enable debug mode. This will notably print exceptions with locals.  [default: no-debug]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `data`: Display the asset manager and file...
* `show`: Display information useful for debugging.
* `srf`: Spectral response function filtering utility.

### `eradiate data`

Display the asset manager and file resolver configurations.
Use subcommands for other data management tasks.

**Usage**:

```console
$ eradiate data [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `clear`: Delete data.
* `download`: Download a resource from remote storage to...
* `install`: Install a resource.
* `list`: List all packages referenced by the...
* `remove`: Uninstall a resource.
* `update`: Download the data registry manifest from...

#### `eradiate data clear`

Delete data.

**Usage**:

```console
$ eradiate data clear [OPTIONS] [RESOURCE_IDS]...
```

**Arguments**:

* `[RESOURCE_IDS]...`: Resource(s) for which to clear data. If unset, all data will be wiped.

**Options**:

* `--what [cached|unpacked|installed|all]`: A keyword that specifies what to clear.  [default: cached]
* `--all / --no-all`: Alias to --what all.  [default: no-all]
* `--unpacked / --no-unpacked`: Alias to --what unpacked.  [default: no-unpacked]
* `--installed / --no-installed`: Alias to --what installed.  [default: no-installed]
* `--help`: Show this message and exit.

#### `eradiate data download`

Download a resource from remote storage to the cache directory.

**Usage**:

```console
$ eradiate data download [OPTIONS] RESOURCE_IDS...
```

**Arguments**:

* `RESOURCE_IDS...`: One or multiple resource IDs or aliases.  [required]

**Options**:

* `--unpack / --no-unpack`: Unpack downloaded archives.  [default: unpack]
* `--help`: Show this message and exit.

#### `eradiate data install`

Install a resource. If the data is not already cached locally, it is
downloaded from remove storage.

**Usage**:

```console
$ eradiate data install [OPTIONS] RESOURCE_IDS...
```

**Arguments**:

* `RESOURCE_IDS...`: One or multiple resource IDs or aliases.  [required]

**Options**:

* `--help`: Show this message and exit.

#### `eradiate data list`

List all packages referenced by the manifest and their current state
(cached, unpacked, installed).

**Usage**:

```console
$ eradiate data list [OPTIONS]
```

**Options**:

* `--what [resources|aliases|all]`: A keyword that specifies what to clear.  [default: resources]
* `--aliases / --no-aliases`: Alias to --what aliases.  [default: no-aliases]
* `--all / --no-all`: Alias to --what all.  [default: no-all]
* `--help`: Show this message and exit.

#### `eradiate data remove`

Uninstall a resource.

**Usage**:

```console
$ eradiate data remove [OPTIONS] RESOURCE_IDS...
```

**Arguments**:

* `RESOURCE_IDS...`: One or multiple resource IDs or aliases.  [required]

**Options**:

* `--help`: Show this message and exit.

#### `eradiate data update`

Download the data registry manifest from the remote data location.

**Usage**:

```console
$ eradiate data update [OPTIONS]
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
