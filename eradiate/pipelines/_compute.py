import typing as t

import attr
import numpy as np

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..units import symbol


@parse_docs
@attr.s
class ComputeReflectance(PipelineStep):
    """
    Derive reflectance from radiance and irradiance values.
    """

    radiance_var: str = documented(
        attr.ib(default="radiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing leaving radiance values.",
        type="str",
        default='"radiance"',
    )

    irradiance_var: str = documented(
        attr.ib(default="irradiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing irradiance (incoming flux) values.",
        type="str",
        default='"irradiance"',
    )

    brdf_var: str = documented(
        attr.ib(default="brdf", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing BRDF values.",
        type="str",
        default='"brdf"',
    )

    brf_var: str = documented(
        attr.ib(default="brf", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing BRF values.",
        type="str",
        default='"brf"',
    )

    def transform(self, x: t.Any) -> t.Any:
        # Compute BRDF and BRF
        result = x.copy()

        # We assume that all quantities are stored in kernel units
        result[self.brdf_var] = result[self.radiance_var] / result[self.irradiance_var]
        result[self.brdf_var].attrs = {
            "standard_name": "brdf",
            "long_name": "bi-directional reflection distribution function",
            "units": symbol("1/sr"),
        }

        result[self.brf_var] = result[self.brdf_var] * np.pi
        result[self.brf_var].attrs = {
            "standard_name": "brf",
            "long_name": "bi-directional reflectance factor",
            "units": symbol("dimensionless"),
        }

        return result


@parse_docs
@attr.s
class ComputeAlbedo(PipelineStep):
    """
    Derive the albedo from radiosity and irradiance fields.
    """

    radiosity_var: str = documented(
        attr.ib(default="radiosity", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the radiosity (leaving flux) value.",
        type="str",
        default='"radiosity"',
    )

    irradiance_var: str = documented(
        attr.ib(default="irradiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the irradiance (incoming flux) value.",
        type="str",
        default='"irradiance"',
    )

    albedo_var: str = documented(
        attr.ib(default="albedo", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the albedo value.",
        type="str",
        default='"albedo"',
    )

    def transform(self, x: t.Any) -> t.Any:
        # Compute albedo
        result = x.copy()

        # We assume that all quantities are stored in kernel units
        result[self.albedo_var] = (
            result[self.radiosity_var] / result[self.irradiance_var]
        )
        result[self.albedo_var].attrs = {
            "standard_name": "albedo",
            "long_name": "surface albedo",
            "units": "",
        }

        return result
