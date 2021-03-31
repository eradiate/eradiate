import xarray as xr


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
    values_pos = hsphere_dataarray.sel(
        **{
            phi_dim: phi,
            theta_dim: theta_pos,
            "method": "nearest",
        }
    )

    # Retrieve values for negative half-plane
    theta_neg = hsphere_dataarray.coords[theta_dim][1:]
    values_neg = hsphere_dataarray.sel(
        **{
            phi_dim: (phi + 180.0) % 360.0,
            theta_dim: theta_neg,
            "method": "nearest",
        }
    )

    # Transform zeniths to negative values
    values_neg = values_neg.assign_coords({theta_dim: -theta_neg})
    # Reorder data
    values_neg = values_neg.loc[
        {theta_dim: sorted(values_neg.coords[theta_dim].values)}
    ]

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
