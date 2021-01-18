""":mod:`xarray`-related components."""

import attr
import cerberus
import xarray as xr

from . import plot


# -- Metadata processing -------------------------------------------------------

def validate_metadata(data, spec, normalize=False, allow_unknown=False):
    """Validate (and possibly normalise) metadata fields of the ``data``
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
    def schema(self):
        """Cerberus schema for metadata validation and normalisation. This
        default implementation raises a NotImplementedError."""
        raise NotImplementedError


@attr.s
class CoordSpec(DataSpec):
    """Specification for a coordinate variable.

    .. rubric:: Constructor arguments / instance attributes

    ``standard_name`` (str):
        `Standard name as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#standard-name>`_.

    ``units`` (str or None):
        `Units as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units>`_.
        If set to ``None``, the coordinate variable is considered unitless (not
        to be confused with dimensionless), *i.e.* it should be applied no unit.
        This is typically useful for coordinates consisting of string labels,
        which do not have units and are not used for computation.

    ``long_name`` (str):
        `Long name as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#long-name>`_.
    """

    standard_name = attr.ib()
    units = attr.ib()
    long_name = attr.ib()

    @property
    def schema(self):
        """Cerberus schema for metadata validation and normalisation. The
        generated schema can be found in the source code."""
        result = {
            "standard_name": {
                "allowed": [self.standard_name],
                "default": self.standard_name,
                "empty": False,
                "required": True
            },
            "long_name": {
                "allowed": [self.long_name],
                "default": self.long_name,
                "empty": False,
                "required": True
            }
        }

        if self.units is not None:
            result["units"] = {
                "allowed": [self.units],
                "default": self.units,
                "required": True
            }

        return result


class CoordSpecRegistry:
    """Storage for coordinate specifications (:class:`CoordSpec` instances).
    This class also serves stored objects, either individually or as consistent
    collections.

    .. note::

       This class should not be instantiated.
    """

    #: Coordinate specification registry (dict[str, :class:`CoordSpec`]).
    registry = {}

    #: Coordination specification collection registry
    #: (dict[str, dict[str, :class:`CoordSpec`]]).
    registry_collections = {}

    @classmethod
    def register(cls, spec_id, coord_spec):
        """Add a :class:`CoordSpec` instance to the registry.

        Parameter ``spec_id`` (str):
            Registry keyword.

        Parameter ``coord_spec`` (:class:`CoordSpec`):
            Registered object.
        """
        cls.registry[spec_id] = coord_spec

    @classmethod
    def register_collection(cls, collection_id, coord_spec_ids):
        """Add a collection of :class:`CoordSpec` instance to the registry.
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
    def get(cls, coord_spec_id):
        """Query the registry for a coordinate specification.

        Parameter ``coord_spec_id`` (str):
            Coordinate specification identifier to lookup in the registry.

        Returns → :class:`CoordSpec`:
            Looked up coordinate specification.
        """
        return cls.registry[coord_spec_id]

    @classmethod
    def get_collection(cls, collection_id):
        """Query the coolection registry for a coordinate specification
        collection.

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


def _coord_specs_validator(instance, attribute, value):
    for val in value.values():
        if not isinstance(val, CoordSpec):
            raise TypeError(f"while validating {attribute.name}: "
                            f"must be a dict[str, CoordSpec]")


@attr.s
class VarSpec:
    """Specification for a data variable.

    .. rubric:: Constructor arguments / instance attributes

    ``standard_name`` (str or None):
        `Standard name as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#standard-name>`_.
        If ``None``, ``long_name`` must be ``None``. If not ``None``,
        ``long_name`` must not be ``None``.

    ``units`` (str or None):
        `Units as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units>`_.
        If set to ``None``, the coordinate variable is considered unitless (not
        to be confused with dimensionless), *i.e.* it should be applied no unit.
        This is typically useful for coordinates consisting of string labels,
        which do not have units and are not used for computation.

    ``long_name`` (str or None):
        `Long name as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#long-name>`_.
        If ``None``, ``standard_name`` must be ``None``. If not ``None``,
        ``standard_name`` must not be ``None``.

    .. warning::

       Either both or none of ``standard_name`` and ``long_name`` must be
       ``None``.

    ``coord_specs`` (str or dict[str, :class:`CoordSpec`]):
        Specification for variable coordinates. Empty dicts are allowed.
        If a string is passed, it will be converted to a coordinate
        specification collection using
        :meth:`CoordSpecRegistry.str_to_collection`.
    """
    standard_name = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    units = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    long_name = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    coord_specs = attr.ib(
        converter=CoordSpecRegistry.str_to_collection,
        factory=dict,
        validator=[attr.validators.instance_of(dict), _coord_specs_validator]
    )

    def __attrs_post_init__(self):
        args = (self.standard_name, self.long_name)
        if not (all([x is None for x in args]) or
                all([x is not None for x in args])):
            raise ValueError("either all or none of 'standard_name' and "
                             "'long_name' must be None")

    @property
    def dims(self):
        """Return list of dimension coordinate names."""
        return list(self.coord_specs.keys())

    @property
    def schema(self):
        """Cerberus schema for metadata validation and normalisation. The
        generated schema can be found in the source code."""
        result = {
            "standard_name": {
                "allowed": [self.standard_name],
                "default": self.standard_name,
                "empty": False,
                "required": True
            },
            "long_name": {
                "allowed": [self.long_name],
                "default": self.long_name,
                "empty": False,
                "required": True
            }
        }

        if self.units is not None:
            result["units"] = {
                "allowed": [self.units],
                "default": self.units,
                "required": True
            }

        return result


def _var_specs_validator(instance, attribute, value):
    for val in value.values():
        if not isinstance(val, VarSpec):
            raise TypeError(f"while validating {attribute.name}: "
                            f"must be a dict[str, VarSpec]")


@attr.s
class DatasetSpec:
    """Specification for a dataset.

    .. rubric:: Constructor arguments / instance attributes

    ``convention`` (str or None):
        `Convention used for metadata <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_overview>`_.
        Usually set to ``"CF-1.8".``

    ``title`` (str or None):
        `Dataset title as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.

    ``history`` (str or None):
        `Dataset history as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.

    ``source`` (str or None):
        `Dataset production method as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.

    ``references`` (str or None):
        `References describing the data or its production process as implied by the CF-convention <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_.

    .. warning::

       Either all or none of ``convention``, ``title``, ``history``, ``source``
       and ``references`` must be ``None``.

    ``var_specs`` (dict[str, :class:`CoordSpec`]):
        Specification for data variables. Empty dicts are allowed.

    ``coord_specs`` (str or dict[str, :class:`CoordSpec`]):
        Specification for variable coordinates. Empty dicts are allowed.
        If a string is passed, it will be converted to a coordinate
        specification collection using
        :meth:`CoordSpecRegistry.str_to_collection`.
    """
    convention = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    title = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    history = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    source = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    references = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    var_specs = attr.ib(
        factory=dict,
        validator=[attr.validators.instance_of(dict), _var_specs_validator]
    )
    coord_specs = attr.ib(
        converter=CoordSpecRegistry.str_to_collection,
        factory=dict,
        validator=[attr.validators.instance_of(dict), _coord_specs_validator]
    )

    def __attrs_post_init__(self):
        args = {
            "convention": self.convention,
            "title": self.title,
            "history": self.history,
            "source": self.source,
            "references": self.references
        }

        if not (all([x is None for x in args.values()]) or
                all([x is not None for x in args.values()])):
            raise ValueError("either all or none of 'convention', 'title', "
                             "'history', 'source' and 'references' must be None")

    @property
    def schema(self):
        """Cerberus schema for metadata validation and normalisation. The
        generated schema can be found in the source code."""

        params = {
            "convention": self.convention,
            "title": self.title,
            "history": self.history,
            "source": self.source,
            "references": self.references
        }

        return {
            param: {
                "default": value,
                "type": "string",
                "required": True
            } for param, value in params.items()
        }


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
    ("wavelength", CoordSpec("wavelength", "nm", "wavelength")),
    ("z_layer", CoordSpec("layer_altitude", "m", "layer altitude")),
    ("z_level", CoordSpec("level_altitude", "m", "level altitude")),
    ("species", CoordSpec("species", None, "species")),
    ("w", CoordSpec("wavelength", "nm", "wavelength")),
    ("t", CoordSpec("time", None, "time"))
]:
    CoordSpecRegistry.register(coord_spec_id, coord_spec)

# Register coordinate spec collections
for collection_id, coord_spec_ids in [
    ("angular_intrinsic", ("theta_i", "phi_i", "theta_o", "phi_o", "wavelength")),
    ("angular_observation", ("sza", "saa", "vza", "vaa", "wavelength")),
    ("angular_observation_pplane", ("sza", "saa", "vza", "wavelength")),
    ("atmospheric_profile", ("z_layer", "z_level", "species")),
    ("solar_irradiance_spectrum", ("w", "t"))
]:
    CoordSpecRegistry.register_collection(collection_id, coord_spec_ids)


# Define solar irradiance spectra dataset specifications
ssi_dataset_spec = DatasetSpec(
    title="Untitled solar irradiance spectrum",
    convention="CF-1.8",
    source="Unknown",
    history="Unknown",
    references="Unknown",
    var_specs={
        "ssi": VarSpec(
            standard_name="solar_irradiance_per_unit_wavelength",
            units="W/m^2/nm",
            long_name="solar spectral irradiance"
        )
    },
    coord_specs="solar_irradiance_spectrum"
)


def make_dataarray(data, coords=None, dims=None, var_spec=None):
    """Create a :class:`~xarray.DataArray` with default metadata.

    Parameter ``data``:
        Data forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``coords``:
        Coordinates forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``dims``:
        Dimension names forwarded to the :class:`~xarray.DataArray` constructor.

    Parameter ``var_spec`` (:class:`VarSpec` or None):
        If not ``None``, data variable specification used to apply default
        metadata.
    """

    dataarray = xr.DataArray(data, coords=coords, dims=dims)
    if var_spec is not None:
        dataarray.ert.normalize_metadata(var_spec)

    return dataarray


# -- Angular data selection ----------------------------------------------------

def plane(hsphere_dataarray, phi, theta_dim="theta_o", phi_dim="phi_o", drop=False):
    """Extract a plane data set from a hemispherical data array.
    This method will select data on a plane oriented along the azimuth direction
    ``phi`` and its complementary ``phi`` + 180°, and stitch the two subsets
    together.

    Data at azimuth angle ``phi`` will be mapped to positive zenith values,
    while data at ``phi`` + 180° will be mapped to negative zenith values.

    .. note::

       * By default, this function operates on angle dimensions expected for an
         angular data set of intrinsic type.

       * If ``hsphere_dataarray`` contains other non-angular dimensions
         (*e.g.* wavelength), they will persist in the returned array.

       * Just like when selecting data, ``phi_dim`` will be retained in the
         returned array as a scalar coordinate. If ``drop`` is ``True``, it
         will be dropped.

    Parameter ``hdata`` (:class:`~xarray.DataArray`)
        Data set from which to the create the plane data set.

    Parameter ``phi`` (float)
        Viewing azimuth angle to orient the plane view. If set to None,
        phi will be set to be equal to ``phi_i``, providing the principal plane.

    Parameter ``theta_dim`` (str)
        Zenith angle dimension.

    Parameter ``phi_dim`` (str)
        Azimuth angle dimension.

    Parameter ``drop`` (bool)
        If ``True``, drop azimuth angle dimension instead of making it scalar.

    Returns → :class:`~xarray.DataArray`
        Extracted plane data set for the requested azimuth angle.
    """
    # Retrieve values for positive half-plane
    theta_pos = hsphere_dataarray.coords[theta_dim]
    values_pos = hsphere_dataarray.sel(**{
        phi_dim: phi,
        theta_dim: theta_pos,
        "method": "nearest",
    })

    # Retrieve values for negative half-plane
    theta_neg = hsphere_dataarray.coords[theta_dim][1:]
    values_neg = hsphere_dataarray.sel(**{
        phi_dim: (phi + 180.) % 360.,
        theta_dim: theta_neg,
        "method": "nearest",
    })

    # Transform zeniths to negative values
    values_neg = values_neg.assign_coords({theta_dim: -theta_neg})
    # Reorder data
    values_neg = values_neg.loc[{theta_dim: sorted(values_neg.coords[theta_dim].values)}]

    # Combine negative and positive half-planes; drop the azimuth dimension
    # (inserted as a scalar dimension afterwards)
    result = xr.concat((values_neg, values_pos), dim=theta_dim).drop_vars(phi_dim)
    # We don't forget to copy metadata
    result.coords[theta_dim].attrs = hsphere_dataarray.coords[theta_dim].attrs
    # By convention, we assign to all points the azimuth coordinate of the
    # positive half-plane (and reduce the corresponding dimension to 1)
    if not drop:
        result = result.assign_coords({phi_dim: phi})
        result.coords[phi_dim].attrs = hsphere_dataarray.coords[phi_dim].attrs

    return result


def pplane(bhsphere_dataarray, sza=None, saa=None):
    """Extract a principal plane view from a bi-hemispherical observation data
    set. This operation, in practice, consists in extracting a hemispherical
    view based on the incoming direction angles ``sza`` and ``saa``, then
    applying the plane view extraction function :func:`plane` with
    ``phi = saa``.

    .. note::

       This function **will not work** with an angular data set not following
       the observation angular dimension naming (see
       :ref:`Working with angular data <sec-user_guide-data_guide-working_angular_data>`).

    Parameter ``bhsphere_dataarray`` (:class:`~xarray.DataArray`):
        Bi-hemispherical data array (with four angular directions).

    Parameter ``sza`` (float or None):
        Solar zenith angle. If `None`, select the first value available.

    Parameter ``saa`` (float or None):
        Solar azimuth angle. If `None`, select the first value available.

    Returns → :class:`~xarray.DataArray`
        Extracted principal plane data set for the requested incoming angular
        configuration.
    """
    if sza is None:
        try:  # if sza is an array
            sza = float(bhsphere_dataarray.coords["sza"][0])
        except TypeError:  # if sza is scalar
            sza = float(bhsphere_dataarray.coords["sza"])

    if saa is None:
        try:  # if saa is an array
            saa = float(bhsphere_dataarray.coords["saa"][0])
        except TypeError:  # if saa is scalar
            saa = float(bhsphere_dataarray.coords["saa"])

    hsphere_dataarray = bhsphere_dataarray.sel(sza=sza, saa=saa)
    return plane(hsphere_dataarray, phi=saa, theta_dim="vza", phi_dim="vaa", drop=True)


# -- Accessor ------------------------------------------------------------------

@xr.register_dataarray_accessor("ert")
class EradiateDataArrayAccessor:
    """Convenience wrapper for operations on :class:`~xarray.DataArray`
    instances. Accessed as a ``DataArray.ert`` property."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot_pcolormesh_polar(self, **kwargs):
        """Wraps :func:`.pcolormesh_polar`."""
        return plot.pcolormesh_polar(self._obj, **kwargs)

    def extract_plane(self, phi, theta_dim="theta_o", phi_dim="phi_o"):
        """Wraps :func:`plane`."""
        return plane(self._obj, phi, theta_dim=theta_dim, phi_dim=phi_dim)

    def extract_pplane(self, sza=None, saa=None):
        """Wraps :func:`pplane`."""
        return pplane(self._obj, sza=sza, saa=saa)

    def validate_metadata(self, var_spec, normalize=False, allow_unknown=True):
        """Validate the metadata for the wrapped :class:`~xarray.DataArray`.
        This function wraps :func:`validate_metadata`.

        Parameter ``var_spec`` (:class:`VarSpec`):
            Data variable specification used to validate the data. If
            ``var_spec.standard_name`` is ``None``, no metadata validation
            will be attempted on the variable: only coordinate variable metadata
            validation will be performed.

        Parameter ``normalize`` (bool):
            If ``True``, also normalise metadata in-place.
        """
        dataarray = self._obj

        # Validate variable metadata
        if var_spec.standard_name is not None:
            dataarray.attrs = validate_metadata(
                data=dataarray,
                spec=var_spec,
                normalize=normalize,
                allow_unknown=allow_unknown
            )

        # Validate dimension coordinate metadata
        for dim, coord_spec in var_spec.coord_specs.items():
            dataarray.coords[dim].attrs = validate_metadata(
                data=dataarray.coords[dim],
                spec=coord_spec,
                normalize=normalize,
                allow_unknown=allow_unknown
            )

    def normalize_metadata(self, var_spec):
        """Validate the metadata for the wrapped :class:`~xarray.DataArray`.
        Basically a call to :meth:`validate_metadata` with ``normalize`` set to
        ``True``.
        """
        self.validate_metadata(var_spec, normalize=True, allow_unknown=False)


@xr.register_dataset_accessor("ert")
class EradiateDatasetAccessor:
    """Convenience wrapper for operations on :class:`~xarray.Dataset` instances.
    Accessed as a ``Dataset.ert`` property."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate_metadata(self, dataset_spec, normalize=False, allow_unknown=True):
        """Validate the metadata for the wrapped :class:`~xarray.Dataset`.
        This function wraps :func:`validate_metadata`.

        Parameter ``dataset_spec`` (:class:`DatsetSpec`):
            Dataset specification used to validate the data.
            If ``dataset_spec.title`` is ``None``, no metadata validation
            will be attempted on the dataset: only data and coordinate variable
            metadata validation will be performed.

        Parameter ``normalize`` (bool):
            If ``True``, also normalise metadata in-place.

        Parameter ``allow_unknown`` (bool):
            If ``True``, unknown attributes are allowed.
        """
        dataset = self._obj

        # Validate dataset metadata
        if dataset_spec.title is not None:
            dataset.attrs = validate_metadata(
                data=dataset,
                spec=dataset_spec,
                normalize=normalize,
                allow_unknown=allow_unknown
            )

        # Validate data variable metadata
        for data_var, var_spec in dataset_spec.var_specs.items():
            dataarray = dataset.data_vars[data_var]
            if var_spec.standard_name is not None:
                try:
                    dataarray.attrs = validate_metadata(
                        data=dataarray,
                        spec=var_spec,
                        normalize=normalize,
                        allow_unknown=allow_unknown
                    )
                except ValueError as e:
                    raise ValueError(f"data variable '{data_var}': {str(e)}")

        # Validate dimension coordinate metadata
        for dim, coord_spec in dataset_spec.coord_specs.items():
            try:
                dataset.coords[dim].attrs = validate_metadata(
                    data=dataset.coords[dim],
                    spec=coord_spec,
                    normalize=normalize,
                    allow_unknown=allow_unknown
                )
            except ValueError as e:
                raise ValueError(f"coordinate '{dim}': {str(e)}")

    def normalize_metadata(self, dataset_spec):
        """Validate the metadata for the wrapped :class:`~xarray.Dataset`.
        Basically a call to :meth:`validate_metadata` with ``normalize`` set to
        ``True``.
        """
        self.validate_metadata(dataset_spec, normalize=True, allow_unknown=False)
