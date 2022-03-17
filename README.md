![Eradiate logo](docs/fig/eradiate-logo.svg "Eradiate — A new-generation radiative transfer simulation package")

# Eradiate Radiative Transfer Model

[![docs][rtd-badge]][rtd-url]
[![build][docker-badge]][docker-url]
[![black][black-badge]][black-url]
[![isort][isort-badge]][isort-url]

[rtd-badge]: https://img.shields.io/readthedocs/eradiate?logo=readthedocs&logoColor=white&style=flat-square
[rtd-url]: https://eradiate.readthedocs.io/en/latest/
[docker-badge]: https://img.shields.io/github/workflow/status/eradiate/eradiate/Docker%20build?label=docker&logo=docker&logoColor=white&style=flat-square
[docker-url]: https://github.com/eradiate/eradiate/actions/workflows/docker.yml
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
[black-url]: https://github.com/psf/black/
[isort-badge]: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat-square&labelColor=ef8336
[isort-url]: https://pycqa.github.io/isort/

Eradiate is a modern radiative transfer simulation software package for Earth
observation applications. Its main focus is accuracy, and for that purpose, it
uses the Monte Carlo ray tracing method to solve the radiative transfer
equation.

## Installation and usage

For build and usage instructions, refer to the
[documentation](https://eradiate.readthedocs.org).

## Support

Got a question? Please visit our
[discussion forum](https://github.com/eradiate/eradiate/discussions).

## Authors and acknowledgements

Eradiate is developed by a core team consisting of Vincent Leroy,
Yvan Nollet, Sebastian Schunke, Nicolas Misk and Yves Govaerts.

Eradiate uses the
[Mitsuba 2 renderer](https://github.com/mitsuba-renderer/mitsuba2), developed by
the [Realistic Graphics Lab](https://rgl.epfl.ch/),
taking advantage of its Python interface and proven architecture, and extends it
with components implementing numerical methods and models used in radiative
transfer for Earth observation. The Eradiate team acknowledges Mitsuba creators
and contributors for their work.

The development of Eradiate is funded by the
[Copernicus programme](https://www.copernicus.eu/) through a project managed by
the [European Space Agency](http://www.esa.int/).
The design phase was funded by the [MetEOC-3 project](http://www.meteoc.org/)
(EMPIR grant 16ENV03).

## License

Eradiate is free software licensed under the
[GNU Public License (v3)](./LICENSE).

## Project status

Eradiate is actively developed. It is beta software.
