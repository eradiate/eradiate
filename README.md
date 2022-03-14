![Eradiate logo](docs/fig/eradiate-logo-dark-no_bg.png "Eradiate — A new-generation radiative transfer simulation package")

# Eradiate Radiative Transfer Model

[![docs][1]][2]
[![build][3]][4]

[1]: https://img.shields.io/readthedocs/eradiate?logo=readthedocs&logoColor=white&style=flat-square
[2]: https://eradiate.readthedocs.io/en/latest/
[3]: https://img.shields.io/github/workflow/status/eradiate/eradiate/Docker%20build?label=docker&logo=docker&logoColor=white&style=flat-square
[4]: https://github.com/eradiate/eradiate/actions/workflows/docker.yml

Eradiate is a modern radiative transfer simulation software package for Earth 
observation applications. Its main focus is accuracy, and for that purpose, it
uses the Monte Carlo ray tracing method to solve the radiative transfer 
equation.

## Installation and usage

For build and usage instructions, refer to the 
[documentation](https://eradiate.readthedocs.org).

## Support

Got a question? Please visit our [discussion forum](https://github.com/eradiate/eradiate/discussions).

## Authors and acknowledgements

Eradiate is developed by a core team consisting of Vincent Leroy, 
Yvan Nollet, Sebastian Schunke, Nicolas Misk and Yves Govaerts.

Eradiate uses the [Mitsuba 2 renderer](https://github.com/mitsuba-renderer/mitsuba2)
as its radiometric kernel, taking advantage of its Python interface and proven 
architecture, and extends it with components implementing numerical methods and 
models used in radiative transfer for Earth observation. The Eradiate team 
acknowledges Mitsuba creators and contributors for their exceptional work.

The development of Eradiate is funded by a European Space Agency project 
supported by the European Commission's Copernicus programme. The design phase 
was funded by the MetEOC-3 project.

## License

Eradiate is free software licensed under the [GNU Public License (v3)](./LICENSE).

## Project status

Eradiate is actively developed. It is beta software.
