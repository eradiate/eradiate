![Eradiate logo](docs/_static/eradiate-logo.svg "Eradiate — A new-generation radiative transfer simulation package")

# Eradiate Radiative Transfer Model

[![pypi][pypi-badge]][pypi-url]
[![docs][rtd-badge]][rtd-url]
[![ruff][ruff-badge]][ruff-url]
[![zenodo][zenodo-badge]][zenodo-url]

[pypi-badge]: https://img.shields.io/pypi/v/eradiate?style=flat-square
[pypi-url]: https://pypi.org/project/eradiate/
[rtd-badge]: https://img.shields.io/readthedocs/eradiate?logo=readthedocs&logoColor=white&style=flat-square
[rtd-url]: https://eradiate.readthedocs.io/en/latest/
[ruff-badge]: https://img.shields.io/badge/%E2%9A%A1%EF%B8%8F-ruff-%23171029?style=flat-square
[ruff-url]: https://ruff.rs
[zenodo-badge]: https://img.shields.io/badge/doi-10.5281/zenodo.7224314-blue.svg?style=flat-square
[zenodo-url]: https://zenodo.org/records/7224314

![Development supported by ESA and the European Commission (light mode)](docs/_static/esa_copernicus-logos-light.svg#gh-light-mode-only)

![Development supported by ESA and the European Commission (dark mode)](docs/_static/esa_copernicus-logos-dark.svg#gh-dark-mode-only)

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
  Eradiate ships spectral data in the solar reflective region (at least from
  280 nm to 2500 nm).
  </details>

  <details>
  <summary>
  Line-by-line simulation
  </summary>
  These are true monochromatic simulations (as opposed to narrow band
  simulations).
  Eradiate provides
  <a href="https://eradiate.readthedocs.io/en/stable/data/absorption_databases.html">monochromatic absorption databases</a> covering the [250, 3125] nm interval.
  User-defined absorption databases are also supported (see the
  <a href="https://eradiate.readthedocs.io/en/stable/data/absorption_databases.html#format-monochromatic">database format</a>).
  </details>

  <details>
  <summary>
  Band simulation
  </summary>
  These simulations computes results in spectral bands.
  The correlated <em>k</em>-distribution (CKD) method with configurable
  quadrature rule is used. This method achieves a trade-off between performance
  and accuracy for the simulation of absorption by gases.
  Eradiate provides
  <a href="https://eradiate.readthedocs.io/en/stable/data/absorption_databases.html">CKD-ready absorption databases</a>
  for the [250, 3125] nm
  interval, with various spectral bin sizes (100 cm⁻¹, 1 nm, 10 nm).
  User-defined absorption databases are also supported (see the
  <a href="https://eradiate.readthedocs.io/en/stable/data/absorption_databases.html#format-ckd">database format</a>).
  </details>

  <details>
  <summary>
  Polarization
  </summary>
  Eradiate optionally supports polarized light simulation. This feature can be
  switched on or off to achieve the best compromise between accuracy and
  performance.
  </details>
  </li>

  <li><strong>Atmosphere</strong>

  <details>
  <summary>
  One-dimensional atmospheric profiles
  </summary>
  Both standard profiles, <em>e.g.</em> the AFGL (1986) profiles, and customized
  profiles are supported.
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
  Lambertian, RPV, Ross Thick-Li Sparse, Hapke and ocean surface reflection models
  </summary>
  All models can be parametrized against the spectral dimension.
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
  Directional or finite-size illumination model
  </summary>
  Eradiate supports both ideal (Delta angular distribution), and realistic
  (finite angular size) illumination models.
  </details>

  <details>
  <summary>
  Many irradiance datasets
  </summary>
  <a href="https://eradiate.readthedocs.io/en/stable/data/solar_irradiance.html#available-datasets">Pick your favourite</a>
  or
  <a href="https://eradiate.readthedocs.io/en/stable/data/solar_irradiance.html">bring your own</a>.
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
  Our <a href="https://eradiate.readthedocs.io/en/stable/data/srf.html">SRF data</a>
  is very close to the original data, and we provide advice to
  further clean up the data and find the right balance between accuracy and
  performance.
  </details>
  </li>

  <li><strong>Monte Carlo ray tracing</strong>

  <details>
  <summary>
  Mitsuba renderer as radiometric kernel
  </summary>
  We leverage the advanced Python API of a cutting-edge C++ rendering system.
  </details>

  <details>
  <summary>
  State-of-the-art volumetric path tracing algorithm
  </summary>
  Mitsuba ships a null-collision-based volumetric path tracer which performs
  well in many of the cases Eradiate is used for. We also provide a
  special-purpose path tracing algorithm for plane-parallel geometries that can
  perform up to 2 orders of magnitude faster than the null-collision algorithm.
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
  Eradiate uses predominantly <a href="https://xarray.dev">Xarray</a> data
  structures for I/O.
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

Eradiate is developed by a core team consisting of Vincent Leroy,
Claudia Emde, Nicolae Marton, Nicolas Misk and Yves Govaerts. For more
information about the Eradiate team,
[visit our website](https://www.eradiate.eu/site/people).

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

## Citing Eradiate

The most general citation is as follows:

```bibtex
@software{Eradiate,
    author = {Leroy, Vincent and Nollet, Yvan and Schunke, Sebastian and Misk, Nicolas and Marton, Nicolae and Govaerts, Yves},
    license = {LGPL-3.0},
    title = {Eradiate radiative transfer model},
    url = {https://github.com/eradiate/eradiate},
    doi = {10.5281/zenodo.7224314},
    year = {2024}
}
```

If you want to reference a specific version, you can update the previous
citation with `doi`, `year` and `version` fields populated with metadata
retrieved from our
[Zenodo records](https://zenodo.org/search?q=parent.id%3A7224314&f=allversions%3Atrue&l=list&p=1&s=10&sort=version).
Example:

```bibtex
@software{Eradiate,
    author = {Leroy, Vincent and Nollet, Yvan and Schunke, Sebastian and Misk, Nicolas and Marton, Nicolae and Govaerts, Yves},
    license = {LGPL-3.0},
    title = {Eradiate radiative transfer model},
    url = {https://github.com/eradiate/eradiate},
    doi = {10.5281/zenodo.13897261},
    year = {2024},
    version = {0.29.0},
}
```

## License

Eradiate is free software licensed under the
[GNU Lesser General Public License (v3)](./LICENSE).

## Project status

Eradiate is actively developed. It is beta software.
