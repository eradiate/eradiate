__all__ = [
    "CoordSpec",
    "DatasetSpec",
    "VarSpec",
    "validate_metadata",
]

from typing import Dict, List, Optional, Union

import attr
import cerberus
import xarray

from eradiate._attrs import documented, parse_docs


def validate_metadata(
    data: Union[xarray.Dataset, xarray.DataArray, xarray.Coordinate],
    spec: "DataSpec",
    normalize: bool = False,
    allow_unknown: bool = False,
):
    """
    Validate (and possibly normalise) metadata fields of the ``data``
    parameter based on a data specification.

    Parameter ``data``:
        Either a dataset, data array / data variable or coordinate variable to
        validate the metadata of.

    Parameter ``spec`` (:class:`DataSpec`):
        Appropriate :class:`DataSpec` child class matching the type of ``data``.

    Parameter ``normalize`` (bool):
        If ``True``, also normalise metadata.

    Parameter ``allow_unknown`` (bool):
        If ``True``, allows unknown keys.

    Returns:
        If ``normalize`` is ``True``, normalised metadata dictionary; otherwise,
        unmodified metadata dictionary.

    Raises → ValueError:
        Got errors during metadata validation.
    """
    v = cerberus.Validator(schema=spec.schema, allow_unknown=allow_unknown)
    v.validate(data.attrs, normalize=normalize)
    if not v.errors:
        if normalize:
            return v.document
        else:
            return data.attrs
    else:
        raise ValueError(f"while validating metadata, got errors {v.errors}")


@attr.s
class DataSpec:
    """Interface for data specification classes."""

    @property
    def schema(self) -> Dict:
        """Cerberus schema for metadata validation and normalisation. This
        default implementation raises a NotImplementedError."""
        raise NotImplementedError


@parse_docs
@attr.s
class CoordSpec(DataSpec):
    """Specification for a coordinate variable."""

    standard_name: Optional[str] = documented(
        attr.ib(),
        doc="`Standard name as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#standard-name>`_.",
        type="str",
    )

    units: Optional[str] = documented(
        attr.ib(),
        doc="`Units as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units>`_. "
        "If set to ``None``, the coordinate variable is considered unitless (not "
        "to be confused with dimensionless), *i.e.* it should be applied no unit. "
        "This is typically useful for coordinates consisting of string labels, "
        "which do not have units and are not used for computation. ",
        type="str or None",
    )

    long_name: Optional[str] = documented(
        attr.ib(),
        doc="`Long name as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#long-name>`_.",
        type="str",
    )

    @property
    def schema(self) -> Dict:
        """
        Cerberus schema for metadata validation and normalisation. The generated
        schema can be found in the source code.
        """
        result = {
            "standard_name": {
                "allowed": [self.standard_name],
                "default": self.standard_name,
                "empty": False,
                "required": True,
            },
            "long_name": {
                "allowed": [self.long_name],
                "default": self.long_name,
                "empty": False,
                "required": True,
            },
        }

        if self.units is not None:
            result["units"] = {
                "allowed": [self.units],
                "default": self.units,
                "required": True,
            }

        return result


class CoordSpecRegistry:
    """
    Storage for coordinate specifications (:class:`CoordSpec` instances).
    This class also serves stored objects, either individually or as consistent
    collections.

    .. note:: This class should not be instantiated.
    """

    #: Coordinate specification registry (dict[str, :class:`CoordSpec`]).
    registry: Dict = {}

    #: Coordination specification collection registry
    #: (dict[str, dict[str, :class:`CoordSpec`]]).
    registry_collections: Dict = {}

    @classmethod
    def register(cls, spec_id: str, coord_spec: CoordSpec):
        """
        Add a :class:`CoordSpec` instance to the registry.

        Parameter ``spec_id`` (str):
            Registry keyword.

        Parameter ``coord_spec`` (:class:`CoordSpec`):
            Registered object.
        """
        cls.registry[spec_id] = coord_spec

    @classmethod
    def register_collection(cls, collection_id: str, coord_spec_ids: List[str]):
        """
        Add a collection of :class:`CoordSpec` instance to the registry.
        Registered coordinate specifications must be already registered
        to this registry (see :meth:`register`).

        Parameter ``collection_id`` (str):
            Registry keyword.

        Parameter ``coord_spec_ids`` (list[str]):
            List of coordinate specification registry keys to add to the created
            collection.
        """
        cls.registry_collections[collection_id] = {
            spec_id: cls.registry[spec_id] for spec_id in coord_spec_ids
        }

    @classmethod
    def get(cls, coord_spec_id: str) -> CoordSpec:
        """Query the registry for a coordinate specification.

        Parameter ``coord_spec_id`` (str):
            Coordinate specification identifier to lookup in the registry.

        Returns → :class:`CoordSpec`:
            Looked up coordinate specification.
        """
        return cls.registry[coord_spec_id]

    @classmethod
    def get_collection(cls, collection_id: str) -> Dict[str, CoordSpec]:
        """
        Query the collection registry for a coordinate specification collection.

        Parameter ``collection_id`` (str):
            Coordinate specification collection identifier to lookup in the
            registry.

        Returns → dict[str, :class:`CoordSpec`]:
            Looked up coordinate specification collection.
        """
        return cls.registry_collections[collection_id]

    @classmethod
    def str_to_collection(cls, x):
        """Attempt conversion of ``x`` to a coordinate specification collection.
        This class method is intended for use as an ``attrs`` converter.

        Parameter ``x``:
            Object to attempt conversion of.

        Returns:
            If ``x`` is a string, the result of :meth:`get_collection` is
            returned. Otherwise, ``x`` is returned unchanged.
        """
        if isinstance(x, str):
            return cls.get_collection(x)
        return x


# Register coordinate specs
for coord_spec_id, coord_spec in [
    ("theta_o", CoordSpec("outgoing_zenith_angle", "deg", "outgoing zenith angle")),
    ("phi_o", CoordSpec("outgoing_azimuth_angle", "deg", "outgoing azimuth angle")),
    ("theta_i", CoordSpec("incoming_zenith_angle", "deg", "incoming zenith angle")),
    ("phi_i", CoordSpec("incoming_azimuth_angle", "deg", "incoming azimuth angle")),
    ("vza", CoordSpec("viewing_zenith_angle", "deg", "viewing zenith angle")),
    ("vaa", CoordSpec("viewing_azimuth_angle", "deg", "viewing azimuth angle")),
    ("sza", CoordSpec("solar_zenith_angle", "deg", "solar zenith angle")),
    ("saa", CoordSpec("solar_azimuth_angle", "deg", "solar azimuth angle")),
    ("z_layer", CoordSpec("layer_altitude", "m", "layer altitude")),
    ("z_level", CoordSpec("level_altitude", "m", "level altitude")),
    ("species", CoordSpec("species", None, "species")),
    ("w", CoordSpec("wavelength", "nanometer", "wavelength")),
    ("t", CoordSpec("time", None, "time")),
]:
    CoordSpecRegistry.register(coord_spec_id, coord_spec)

# Register coordinate spec collections
for collection_id, coord_spec_ids in [
    ("angular_intrinsic", ("theta_i", "phi_i", "theta_o", "phi_o", "w")),
    ("angular_observation", ("sza", "saa", "vza", "vaa", "w")),
    ("angular_observation_pplane", ("sza", "saa", "vza", "w")),
    ("atmospheric_profile", ("z_layer", "z_level", "species")),
    ("solar_irradiance_spectrum", ("w", "t")),
]:
    CoordSpecRegistry.register_collection(collection_id, coord_spec_ids)


def _coord_specs_validator(instance, attribute, value):
    for val in value.values():
        if not isinstance(val, CoordSpec):
            raise TypeError(
                f"while validating {attribute.name}: " f"must be a dict[str, CoordSpec]"
            )


@parse_docs
@attr.s
class VarSpec:
    """Specification for a data variable."""

    standard_name: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Standard name as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#standard-name>`_. "
        "If ``None``, ``long_name`` must be ``None``. If not ``None``, "
        "``long_name`` must not be ``None``.",
        type="str or None",
    )

    units: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Units as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units>`_. "
        "If set to ``None``, the coordinate variable is considered unitless (not "
        "to be confused with dimensionless), *i.e.* it should be applied no unit. "
        "This is typically useful for coordinates consisting of string labels, "
        "which do not have units and are not used for computation.",
        type="str or None",
    )

    long_name: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Long name as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#long-name>`_. "
        "If ``None``, ``standard_name`` must be ``None``. If not ``None``, "
        "``standard_name`` must not be ``None``.\n"
        "\n"
        ".. warning:: Either both or none of ``standard_name`` and "
        "``long_name`` must be ``None``.",
        type="str or None",
    )

    coord_specs: Dict[str, CoordSpec] = documented(
        attr.ib(
            converter=CoordSpecRegistry.str_to_collection,
            factory=dict,
            validator=[attr.validators.instance_of(dict), _coord_specs_validator],
        ),
        doc="Specification for variable coordinates. Empty dicts are allowed. "
        "If a string is passed, it will be converted to a coordinate "
        "specification collection using "
        ":meth:`CoordSpecRegistry.str_to_collection`.",
        type="str or dict[str, :class:`CoordSpec`]",
    )

    def __attrs_post_init__(self):
        args = (self.standard_name, self.long_name)
        if not (all([x is None for x in args]) or all([x is not None for x in args])):
            raise ValueError(
                "either all or none of 'standard_name' and " "'long_name' must be None"
            )

    @property
    def dims(self) -> List[str]:
        """Return list of dimension coordinate names."""
        return list(self.coord_specs.keys())

    @property
    def schema(self) -> Dict:
        """Cerberus schema for metadata validation and normalisation. The
        generated schema can be found in the source code."""
        result = {
            "standard_name": {
                "allowed": [self.standard_name],
                "default": self.standard_name,
                "empty": False,
                "required": True,
            },
            "long_name": {
                "allowed": [self.long_name],
                "default": self.long_name,
                "empty": False,
                "required": True,
            },
        }

        if self.units is not None:
            result["units"] = {
                "allowed": [self.units],
                "default": self.units,
                "required": True,
            }

        return result


def _var_specs_validator(instance, attribute, value):
    for val in value.values():
        if not isinstance(val, VarSpec):
            raise TypeError(
                f"while validating {attribute.name}: " f"must be a dict[str, VarSpec]"
            )


@parse_docs
@attr.s
class DatasetSpec:
    """
    Specification for a dataset.

    .. warning:: Either all or none of ``convention``, ``title``, ``history``,
       ``source`` and ``references`` must be ``None``.
    """

    convention: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Convention used for metadata "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_overview>`_. "
        'Usually set to ``"CF-1.8".``',
        type="str or None",
        default="None",
    )

    title: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Dataset title as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.",
        type="str or None",
        default="None",
    )

    history: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Dataset history as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.",
        type="str or None",
        default="None",
    )

    source: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`Dataset production method as implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.",
        type="str or None",
        default="None",
    )

    references: Optional[str] = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="`References describing the data or its production process as "
        "implied by the CF-convention "
        "<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.",
        type="str or None",
        default="None",
    )

    var_specs: Dict[str, CoordSpec] = documented(
        attr.ib(
            factory=dict,
            validator=[attr.validators.instance_of(dict), _var_specs_validator],
        ),
        doc="Specification for data variables. Empty dicts are allowed.",
        type="dict[str, :class:`CoordSpec`]",
        default="dict()",
    )

    coord_specs: Dict[str, CoordSpec] = documented(
        attr.ib(
            converter=CoordSpecRegistry.str_to_collection,
            factory=dict,
            validator=[attr.validators.instance_of(dict), _coord_specs_validator],
        ),
        doc="Specification for variable coordinates. Empty dicts are allowed. "
        "If a string is passed, it will be converted to a coordinate "
        "specification collection using "
        ":meth:`CoordSpecRegistry.str_to_collection`.",
        type="str or dict[str, :class:`CoordSpec`]",
        default="dict()",
    )

    def __attrs_post_init__(self):
        args = {
            "convention": self.convention,
            "title": self.title,
            "history": self.history,
            "source": self.source,
            "references": self.references,
        }

        if not (
            all([x is None for x in args.values()])
            or all([x is not None for x in args.values()])
        ):
            raise ValueError(
                "either all or none of 'convention', 'title', "
                "'history', 'source' and 'references' must be None"
            )

    @property
    def schema(self) -> Dict:
        """Cerberus schema for metadata validation and normalisation. The
        generated schema can be found in the source code."""

        params = {
            "convention": self.convention,
            "title": self.title,
            "history": self.history,
            "source": self.source,
            "references": self.references,
        }

        return {
            param: {"default": value, "type": "string", "required": True}
            for param, value in params.items()
        }
