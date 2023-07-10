![Eradiate logo](docs/fig/eradiate-logo.svg "Eradiate — A new-generation radiative transfer simulation package")

# Eradiate Radiative Transfer Model

[![pypi][pypi-badge]][pypi-url]
[![docs][rtd-badge]][rtd-url]
[![black][black-badge]][black-url]
[![ruff][ruff-badge]][ruff-url]

[pypi-badge]: https://img.shields.io/pypi/v/eradiate?style=flat-square
[pypi-url]: https://pypi.org/project/eradiate/
[rtd-badge]: https://img.shields.io/readthedocs/eradiate?logo=readthedocs&logoColor=white&style=flat-square
[rtd-url]: https://eradiate.readthedocs.io/en/latest/
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
[black-url]: https://github.com/psf/black/
[ruff-badge]: https://img.shields.io/badge/%E2%9A%A1%EF%B8%8F-ruff-red?style=flat-square
[ruff-url]: https://ruff.rs

Eradiate is a modern radiative transfer simulation software package for Earth
observation applications. Its main focus is accuracy, and for that purpose, it
uses the Monte Carlo ray tracing method to solve the radiative transfer
equation.

## Detailed list of features

<ul>
  <li><strong>Spectral computation</strong>

  <details>
  <summary>
  Solar reflective spectral region
  </summary>
  Eradiate ships spectral data within from 280 nm to 2400 nm. This range can be
  extended with additional data (just ask for it!).
  </details>

  <details>
  <summary>
  Line-by-line simulation
  </summary>
  These are true monochromatic simulations (as opposed to narrow band
  simulations).
  </details>

  <details>
  <summary>
  Correlated <em>k</em>-distribution band model (1 nm and 10 nm resolution)
  </summary>
  This method achieves compromise between performance and accuracy for the
  simulation of absorption by gases.
  </details>
  </li>

  <li><strong>Atmosphere</strong>

  <details>
  <summary>
  One-dimensional atmospheric profiles (AFGL atmospheric constituent
  profiles)
  </summary>
  These profiles are available in CKD mode only (the monochromatic mode uses
  the simpler U.S. Standard Atmosphere (1976) model).
  </details>

  <details>
  <summary>
  Plane-parallel and spherical-shell geometries
  </summary>
  This allows for more accurate results at high illumination and viewing
  angles.
  </details>
  </li>

  <li><strong>Surface</strong>

  <details>
  <summary>
  Lambertian and RPV reflection models
  </summary>
  Model parameters can be varied against the spectral dimensions.
  </details>

  <details>
  <summary>
  Detailed surface geometry
  </summary>
  Add a discrete canopy model (either disk-based abstract models, or more
  realistic mesh-based models).
  </details>

  <details>
  <summary>
  Combine with atmospheric profiles
  </summary>
  Your discrete canopy can be integrated within a scene featuring a 1D
  atmosphere model in a fully coupled simulation.
  </details>
  </li>

  <li><strong>Illumination</strong>

  <details>
  <summary>
  Directional illumination model
  </summary>
  An ideal illumination model with a Delta angular distribution.
  </details>

  <details>
  <summary>
  Many irradiance datasets
  </summary>
  Pick your favourite—or bring your own.
  </details>
  </li>

  <li><strong>Measure</strong>

  <details>
  <summary>
  Top-of-atmosphere radiance and BRF computation
  </summary>
  An ideal model suitable for satellite data simulation.
  </details>

  <details>
  <summary>
  Perspective camera sensor
  </summary>
  Greatly facilitates scene setup: inspecting the scene is very easy.
  </details>

  <details>
  <summary>
  Many instrument spectral response functions
  </summary>
  Our SRF data is very close to the original data, and we provide advice to
  further clean up the data, trading off accuracy for performance.
  </details>
  </li>

  <li><strong>Monte Carlo ray tracing</strong>

  <details>
  <summary>
  Mitsuba renderer as radiometric kernel
  </summary>
  We leverage the advanced Python API of this cutting-edge C++ rendering
  library.
  </details>

  <details>
  <summary>
  State-of-the-art volumetric path tracing algorithm
  </summary>
  Mitsuba ships a null-collision-based volumetric path tracer which performs
  well in the cases Eradiate is used for.
  </details>
  </li>

  <li><strong>Traceability</strong>

  <details>
  <summary>
  Documented data and formats
  </summary>
  We explain where our data comes from and how users can build their own data
  in a format compatible with Eradiate's input.
  </details>

  <details>
  <summary>
  Transparent algorithms
  </summary>
  Our algorithms are researched and documented, and their implementation is
  open-source.
  </details>

  <details>
  <summary>
  Thorough testing
  </summary>
  Eradiate is shipped with a large unit testing suite and benchmarked
  periodically against community-established reference simulation software.
  </details>
  </li>

  <li><strong>Interface</strong>

  <details>
  <summary>
  Comprehensive Python interface
  </summary>
  Abstractions are derived from computer graphics and Earth observation and
  are designed to feel natural to EO scientists.
  </details>

  <details>
  <summary>
  Designed for interactive usage
  </summary>
  Jupyter notebooks are now an essential tool in the digital scientific
  workflow.
  </details>

  <details>
  <summary>
  Integration with Python scientific ecosystem
  </summary>
  The implementation is done using the Scientific Python stack.
  </details>

  <details>
  <summary>
  Standard data formats (mostly NetCDF)
  </summary>
  Eradiate uses predominantly xarray data structures for I/O.
  </details>
  </li>
</ul>

## Installation and usage

For build and usage instructions, please refer to the
[documentation](https://eradiate.readthedocs.org).

## Support

Got a question? Please visit our
[discussion forum](https://github.com/eradiate/eradiate/discussions).

## Authors and acknowledgements

Eradiate is developed by a core team consisting of Vincent Leroy, Yvan Nollet,
Sebastian Schunke, Nicolas Misk and Yves Govaerts.

Eradiate uses the
[Mitsuba 3 renderer](https://github.com/mitsuba-renderer/mitsuba3), developed by
the [Realistic Graphics Lab](https://rgl.epfl.ch/),
taking advantage of its Python interface and proven architecture, and extends it
with components implementing numerical methods and models used in radiative
transfer for Earth observation. The Eradiate team acknowledges Mitsuba creators
and contributors for their work.

The development of Eradiate is funded by the
[Copernicus programme](https://www.copernicus.eu/) through a project managed by
the [European Space Agency](http://www.esa.int/) (contract no
40000127201/19/I‑BG).
The design phase was funded by the [MetEOC-3 project](http://www.meteoc.org/)
(EMPIR grant 16ENV03).

## License

Eradiate is free software licensed under the
[GNU Lesser General Public License (v3)](./LICENSE).

## Project status

Eradiate is actively developed. It is beta software.
