# Eradiate Radiative Transfer Model

![Eradiate logo](docs/fig/eradiate-logo-dark-no_bg.png "Eradiate — A new-generation radiative transfer simulation package")

**TODO:** Add badges.

**TODO:** Add software package and project description.

## Documentation

Eradiate's documentation can currently only be browsed offline and has to be built from project sources.


1. Clone the source repository
2. Navigate to the source repository's root directory.
3. Create a build directory, *e.g.*
   ```
   $ mkdir build
   ```
4. Set up the conda environment and set the relevant environment variables
    ```
    $ source conda_create_env.sh
    $ conda activate eradiate
    $ source setpath.sh
    ```
6. Generate the documentation using using Sphinx:
   ```
   $ python -m sphinx html docs build/html
   ```
7. The documentation is generated a `html` subdirectory and can be viewed in a browser

## Building

For build instructions, refer to the [documentation](#documentation). 

## About

Eradiate's development is funded by a European Space Agency project funded by the European Commission's Copernicus programme. The design phase was funded by the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy, Yvan Nollet and Sebastian Schunke.

Eradiate uses the [Mitsuba 2](https://github.com/mitsuba-renderer/mitsuba2) renderer as its computational kernel. The Eradiate team acknowledges Mitsuba 2 creators and contributors for their exceptional work.