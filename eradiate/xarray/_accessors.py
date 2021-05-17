import xarray as xr

from .metadata import validate_metadata


@xr.register_dataarray_accessor("ert")
class EradiateDataArrayAccessor:
    """Convenience wrapper for operations on :class:`~xarray.DataArray`
    instances. Accessed as a ``DataArray.ert`` property."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

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
                allow_unknown=allow_unknown,
            )

        # Validate dimension coordinate metadata
        for dim, coord_spec in var_spec.coord_specs.items():
            dataarray.coords[dim].attrs = validate_metadata(
                data=dataarray.coords[dim],
                spec=coord_spec,
                normalize=normalize,
                allow_unknown=allow_unknown,
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
                allow_unknown=allow_unknown,
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
                        allow_unknown=allow_unknown,
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
                    allow_unknown=allow_unknown,
                )
            except ValueError as e:
                raise ValueError(f"coordinate '{dim}': {str(e)}")

    def normalize_metadata(self, dataset_spec):
        """Validate the metadata for the wrapped :class:`~xarray.Dataset`.
        Basically a call to :meth:`validate_metadata` with ``normalize`` set to
        ``True``.
        """
        self.validate_metadata(dataset_spec, normalize=True, allow_unknown=False)
