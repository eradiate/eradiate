"""Spectral response function (SRF) filtering algorithms."""
import datetime
import typing as t
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pint
import rich
import xarray as xr
from rich.prompt import Confirm
from rich.table import Table

from eradiate import __version__

from . import converters
from .typing import PathLike
from .units import to_quantity


def update_attrs(srf: xr.Dataset, filter_name: str, filter_attr: str) -> None:
    """
    Update data set attributes to indicate that a filtering operation occurred.

    Parameters
    ----------
    srf: Dataset
        Data set whose attributes to update.

    filter_name: str
        Filter name.

    filter_attr: str
        Content of the 'filter' attribute.

    Notes
    -----
    * The update is done inplace.
    * If the dataset has no ``filter`` attribute, it is created. Otherwise, a new line is
      appended to it.
    """
    # filter attribute
    _value = srf.attrs.get("filter")
    previous_filter = _value + "\n" if _value is not None else ""

    # history attribute
    previous_history = srf.attrs["history"] + "\n"
    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    author = f"eradiate {__version__}"
    history_attr = f"{utcnow} - data set filtering ({filter_name}) - {author}"

    # update attributes
    srf.attrs.update(
        {
            "history": f"{previous_history}{history_attr}",
            "filter": f"{previous_filter}{filter_attr}",
        }
    )


def filtering_summary(
    srf: t.Union[PathLike, xr.Dataset], filtered: xr.Dataset
) -> t.Mapping[str, str]:
    srf = converters.load_dataset(srf)
    ni = srf.w.size
    nf = filtered.w.size
    w_min_i = to_quantity(srf.w).min()
    w_max_i = to_quantity(srf.w).max()
    w_min_f = to_quantity(filtered.w).min()
    w_max_f = to_quantity(filtered.w).max()

    return {
        "Initial": {
            "Lower wavelength": f"{w_min_i:~}",
            "Upper wavelength": f"{w_max_i:~}",
            "# wavelength": f"{ni}",
        },
        "Final": {
            "Lower wavelength": f"{w_min_f:~}",
            "Upper wavelength": f"{w_max_f:~}",
            "# wavelength": f"{nf}",
        },
        "Difference": {
            "Lower wavelength": f"{(w_min_f-w_min_i):~}",
            "Upper wavelength": f"{(w_max_f-w_max_i):~}",
            "# wavelength": f"{nf - ni}",
        },
    }


def summarize(
    srf: t.Union[PathLike, xr.Dataset],
    filtered: t.Union[PathLike, xr.Dataset],
) -> Table:
    """
    Produce a summary table of the spectral response function filtering operation.

    Parameters
    ----------
    srf: Dataset
        Initial spectral response function data set.

    filtered: Dataset
        Filtered spectral response function data set.

    Returns
    -------
    :class:`~rich.table.Table`
        Summary table.
    """
    # convert inputs
    srf = converters.load_dataset(srf)
    filtered = converters.load_dataset(filtered)

    summary = filtering_summary(srf=srf, filtered=filtered)
    table = Table(
        title="Filtering summary",
        box=rich.box.HEAVY_HEAD,
        show_footer=True,
    )

    columns = {
        "SRF": "left",
        "Lower wavelength": "center",
        "Upper wavelength": "center",
        "# wavelength": "center",
    }

    footer = {
        **{"SRF": "Difference"},
        **{
            name: summary["Difference"][name]
            for name in ["Lower wavelength", "Upper wavelength", "# wavelength"]
        },
    }

    for name, justification in columns.items():
        table.add_column(name, justify=justification, footer=footer[name])

    for key in summary:
        row_items = [key] + list(summary[key].values())

        if key == "Difference":
            pass
        else:
            table.add_row(*row_items)

    return table


def trim(srf: t.Union[PathLike, xr.Dataset]) -> xr.Dataset:
    """
    Trim all leading zeros except last and all trailing zeros except first.

    Parameters
    ----------
    srf: path-like, Dataset
        Data set to trim.

    Returns
    -------
    Dataset
        Trimmed data set.
    """
    ds = converters.load_dataset(srf)

    # trim
    wsize = ds.srf.values.size
    fsize = np.trim_zeros(ds.srf.values, trim="f").size
    bsize = np.trim_zeros(ds.srf.values, trim="b").size
    istart = wsize - fsize - 1 if wsize > fsize else 0
    istop = bsize if bsize < wsize else wsize - 1
    trimmed = ds.isel(w=range(istart, istop + 1))

    # update history attribute
    previous_history = ds.attrs["history"] + "\n"
    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    author = f"eradiate, version {__version__}"
    history_attr = f"{utcnow} - trimmed data set - {author}"
    trimmed.attrs.update({"history": f"{previous_history}{history_attr}"})

    return trimmed


def save(
    ds: xr.Dataset,
    path: PathLike,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Save dataset to disk.

    Parameters
    ----------
    ds: Dataset
        Dataset to save.

    path: path-like
        Path to which to save the dataset.

    verbose: bool
        If ``True``, display where the dataset is saved.

    dry_run: bool
        If ``True``, display where the dataset would be saved but does not save
        it.
    """
    output_path = Path(path).absolute()
    if dry_run:
        rich.print(f"Would write filtered data to {output_path}")
    else:
        if verbose:
            rich.print(f"Writing filtered data to {output_path}")
        ds.to_netcdf(path)


def trim_and_save(
    srf: t.Union[PathLike, xr.Dataset],
    path: PathLike,
    verbose=False,
    interactive: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Wraps around :meth:`trim` to save the filtered dataset.

    Parameters
    ----------
    srf: path-like, Dataset
        Data set to trim.

    verbose: bool
        If ``True``, display a summary of the trimming operation.

    path: path-like
        Path to which to save the filtered dataset.

    interactive: bool
        If ``True``, display a figure illustrating the filtered region and
        prompt for confirmation before proceeding to saving the dataset to the
        disk.

    dry_run: bool
        If ``True``, displays where the trimmed data set would be saved but
        does not save it.

    See Also
    --------
    :meth:`trim`
    """
    # convert inputs
    ds = converters.load_dataset(srf)
    output_path = Path(path).absolute()

    # trim
    trimmed = trim(srf=srf)

    # print trimming summary table
    if verbose:
        table = summarize(srf=ds, filtered=trimmed)
        rich.print(table)

    # save trimmed dataset
    if interactive:
        show(
            ds=ds,
            title=" ".join(
                [srf.attrs["platform"], srf.attrs["instrument"], srf.attrs["band"]]
            ),
            trim_prior=False,
            threshold=None,
            wmin=to_quantity(trimmed.w).min(),
            wmax=to_quantity(trimmed.w).max(),
            percentage=None,
        )

        if Confirm.ask("Save trimmed dataset?"):
            save(
                ds=trimmed,
                path=output_path,
                verbose=verbose,
                dry_run=dry_run,
            )
        else:
            if verbose:
                rich.print("Aborted!")
    else:
        save(
            ds=trimmed,
            path=output_path,
            verbose=verbose,
            dry_run=dry_run,
        )


def spectral_filter(
    srf: t.Union[PathLike, xr.Dataset],
    wmin: t.Optional[pint.Quantity] = None,
    wmax: t.Optional[pint.Quantity] = None,
) -> xr.Dataset:
    """
    Drop points falling out of wavelength range specified by ``wmin`` and ``wmax``.

    Parameters
    ----------
    srf: Dataset
        Spectral response function data set to filter.

    wmin: quantity
        Minimum wavelength.

    wmax: quantity
        Maximum wavelength.

    Returns
    -------
    Dataset
        Filtered data set.
    """
    srf = converters.load_dataset(srf)

    # filter
    w_units = srf.w.attrs["units"]

    if wmin is not None:
        _wmin = wmin.m_as(w_units)
        filtered = srf.where(srf.w >= _wmin, drop=True)
    else:
        filtered = srf
        _wmin = 0

    if wmax is not None:
        _wmax = wmax.m_as(w_units)
        filtered = filtered.where(srf.w <= _wmax, drop=True)
    else:
        _wmax = "∞"

    # update filtered data set attributes
    update_attrs(
        srf=filtered,
        filter_name="spectral filter",
        filter_attr=(
            f"All points in the original data set that fell out of the "
            f"wavelength range [{_wmin}, {_wmax}] {w_units} were dropped."
        ),
    )

    # sanity check
    if filtered.w.size == 0:
        raise ValueError(
            f"Filtering this data set with wmin = {wmin} and wmax = {wmax} "
            f"would result in empty data set."
        )

    return filtered


def threshold_filter(
    srf: t.Union[PathLike, xr.Dataset],
    value: float = 1e-3,
) -> xr.Dataset:
    """
    Drop data points where response is smaller or equal than a threshold value.

    Parameters
    ----------
    srf: Dataset
        Spectral response function data set to filter.

    value: float
        Spectral response threshold value.

    Raises
    ------
    ValueError: if the threshold value is not in [0, 1[.

    Returns
    -------
    Dataset
        Filtered data set.
    """
    # Convert input
    srf = converters.load_dataset(srf)

    # validate input
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"threshold value should be in [0, 1[ (got {value}).")

    # check that filtering does not disconnect the wavelength space
    filter_indices = np.where(srf.srf.values > value)[0]
    consecutive = np.arange(filter_indices[0], filter_indices[0] + filter_indices.size)
    if not np.all(filter_indices == consecutive):
        warnings.warn(
            f"Filtering this data set with threshold value of {value} would "
            "disconnect the wavelength space. You probably do not want that."
        )

    # filter
    filtered = srf.where(srf.srf > value, drop=True)

    # update filtered data set attributes
    update_attrs(
        srf=filtered,
        filter_name="threshold filter",
        filter_attr=(
            f"All points in the original data set where the spectral response "
            f"function evaluated to {value} or smaller were dropped."
        ),
    )

    # sanity check
    if filtered.w.size == 0:
        raise ValueError(
            f"Filtering this data set with threshold value of {value} would "
            f"result in empty data set."
        )

    return filtered


def integral_filter_w_bounds(
    ds: xr.Dataset, percentage: float = 99.0
) -> t.Tuple[float, float]:
    """
    Compute the wavelength bounds for the integral filter.

    The integrated spectral reponse is computed from the cumulative sum of the
    spectral response function interpolated on a regular grid with the
    smallest wavelength step of the initial data set.

    Parameters
    ----------
    ds: Dataset
        Dataset to filter.

    percentage: float
        Keep data that contribute to this percentage of the integrated spectral
        response.

    Returns
    -------
    tuple
        Wavelength bounds.
    """
    # convert input
    ds = converters.load_dataset(ds)

    # interpolate the spectral reponse on a regular wavelength mesh
    dwmin = ds.w.diff(dim="w").values.min()
    wmin = ds.w.values.min()
    wmax = ds.w.values.max()
    wnum = int((wmax - wmin) / dwmin) + 1
    wreg = np.linspace(wmin, wmax, wnum)
    srfreg = ds.srf.interp(w=wreg)

    # compute the cumulative sums of the interpolated spectral response
    cumsum = np.cumsum(srfreg)
    cumsum_max = cumsum.values.max()  # this is the integrated spectral response

    # this is half the fraction of data points to discard:
    halffraction = (1 - (percentage / 100)) / 2

    # wavelength points that contribute to less than half the percentage of the
    # integrated response function, starting from the smallest wavelength values
    w_left = cumsum.where(cumsum < halffraction * cumsum_max, drop=True).w.values

    # wavelength points that contribute to less than half the percentage of the
    # integrated response function, starting from the smallest wavelength values
    w_right = cumsum.where(cumsum > (1 - halffraction) * cumsum_max, drop=True).w.values

    # determine the wavelength bounds to discard these wavelength points that
    # contribute to less than the specified percentage of the integrated
    # spectral response:
    w0 = w_left[-1] if w_left.size > 0 else wmin
    w1 = w_right[0] if w_right.size > 0 else wmax

    return w0, w1


def integral_filter(
    srf: t.Union[PathLike, xr.Dataset],
    percentage: float = 99.0,
) -> xr.Dataset:
    """
    Keep only data that contribute to the integrated spectral response value
    to the amount of the specified percentage.

    The integrated spectral reponse is computed from the cumulative sum of the
    spectral response function interpolated on a regular grid with the
    smallest wavelength step of the initial data set.

    Parameters
    ----------
    ds: Dataset
        Dataset to filter.

    percentage: float
        Keep data that contribute to this percentage of the integrated spectral
        response.

    Raises
    ------
    ValueError: if the percentage value is not in [0, 100].

    Returns
    -------
    Dataset
        Filtered data set.
    """
    # convert input
    srf = converters.load_dataset(srf)

    # validate_percentage
    if percentage < 0.0 or percentage > 100.0:
        raise ValueError(f"value must be within [0, 100.0] (got {percentage})")

    # compute wavelength bounds and filter
    w_bounds = integral_filter_w_bounds(ds=srf, percentage=percentage)
    filtered = srf.where(srf.w >= w_bounds[0], drop=True).where(
        srf.w <= w_bounds[1], drop=True
    )

    # update filtered data set attributes
    update_attrs(
        srf=filtered,
        filter_name="integral filter",
        filter_attr=(
            f"Data points that did not contribute to {percentage} % of the "
            f"integrated spectral reponse were dropped."
        ),
    )

    # sanity check
    if filtered.w.size == 0:
        raise ValueError(
            f"Filtering this data set with {percentage=} "
            f"would result in empty data set."
        )

    return filtered


def show(
    ds: t.Union[PathLike, xr.Dataset],
    title: str,
    trim_prior: bool = True,
    threshold: t.Optional[float] = None,
    wmin: t.Optional[pint.Quantity] = None,
    wmax: t.Optional[pint.Quantity] = None,
    percentage: t.Optional[float] = None,
) -> None:
    """
    Show filtered region on spectral response function plot.

    Parameters
    ----------
    ds: path-like, Dataset
        Spectral response function to be filtered.

    title: str
        Figure title.

    trim_prior: bool
        If ``True``, trim spectral response function prior to filter.

    threshold: float, optional
        Threshold value for the threshold filter.

    wmin: quantity, optional
        Lower wavelength value for the spectral filter.

    wmax: quantity, optional
        Upper wavelength value for the spectral filter.

    percentage: float, optional
        Percentage value for the integral filter.

    Raises
    ------
    ValueError:
        If the threshold value is not in [0, 1[.
    """
    # setup figure
    plt.figure(dpi=100)
    ax = plt.gca()

    plt_params = {
        "lw": 0.6,
        "marker": ".",
        "markersize": 2,
        "yscale": "log",
    }

    # convert input
    ds = converters.load_dataset(ds)

    if trim_prior:
        trimmed = trim(srf=ds)
    else:
        trimmed = ds
        plt_params.update({"yscale": "linear"})

    trimmed.srf.plot(**plt_params)

    if threshold is not None:
        if threshold < 0.0 or threshold >= 1.0:
            raise ValueError(f"threshold value should be in [0, 1[ (got {threshold}).")

        plt.axhline(y=threshold, color="red", lw=0.5)
        # keep region
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=trimmed.srf.values > threshold,
            facecolor="green",
            alpha=0.1,
        )
        plt.axhline(y=threshold, color="red", lw=0.5)
        # drop region
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=trimmed.srf.values <= threshold,
            facecolor="red",
            alpha=0.1,
        )

    if wmin is not None or wmax is not None:

        if wmin is not None:
            _wmin_value = wmin.m_as(ds.w.attrs["units"])
            plt.axvline(x=_wmin_value, color="red", lw=0.5)
            # drop region
            ax.fill_between(
                trimmed.w.values,
                0,
                trimmed.srf.values,
                where=trimmed.w.values < _wmin_value,
                facecolor="red",
                alpha=0.1,
            )
        else:
            _wmin_value = 0.0

        if wmax is not None:
            _wmax_value = wmax.m_as(ds.w.attrs["units"])
            plt.axvline(x=_wmax_value, color="red", lw=0.5)
            # drop region
            ax.fill_between(
                trimmed.w.values,
                0,
                trimmed.srf.values,
                where=trimmed.w.values > _wmax_value,
                facecolor="red",
                alpha=0.1,
            )
        else:
            _wmax_value = np.inf

        # keep region
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=(
                (trimmed.w.values >= _wmin_value) & (trimmed.w.values <= _wmax_value)
            ),
            facecolor="green",
            alpha=0.1,
        )

    if percentage is not None:
        wmin, wmax = integral_filter_w_bounds(ds=ds, percentage=percentage)
        # drop regions
        plt.axvline(x=wmin, color="red", lw=0.5)
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=trimmed.w.values < wmin,
            facecolor="red",
            alpha=0.1,
        )
        plt.axvline(x=wmax, color="red", lw=0.5)
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=trimmed.w.values > wmax,
            facecolor="red",
            alpha=0.1,
        )
        # keep region
        ax.fill_between(
            trimmed.w.values,
            0,
            trimmed.srf.values,
            where=((trimmed.w.values >= wmin) & (trimmed.w.values <= wmax)),
            facecolor="green",
            alpha=0.1,
        )

    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()


def filter(
    srf: t.Union[PathLike, xr.Dataset],
    path: PathLike,
    verbose: bool = False,
    interactive: bool = False,
    dry_run: bool = False,
    trim_prior: bool = True,
    threshold: t.Optional[float] = None,
    wmin: t.Optional[pint.Quantity] = None,
    wmax: t.Optional[pint.Quantity] = None,
    percentage: t.Optional[float] = None,
) -> None:
    """
    Filter a spectral response function data set.

    Parameters
    ----------
    srf: path-like, Dataset
        Spectral response function data set to filter.

    path: path-like
        Path to which to save the filtered data set.

    verbose: bool
        If ``True``, display afiltering summary table and the path to which
        filtered data set is saved.

    interactive: bool
        If ``True``, display a figure emphasizing the filtered region.

    dry_run: bool
        If ``True``, display the path to which the filtered data set would be
        saved, but does not write the data set to the disk.

    trim_prior: bool
        Trim the data set prior to filtering.

    threshold: float, optional
        Threshold value for the threshold filter.
        See :meth:`threshold_filter`.
        If ``None``, disable the threshold filter.

    wmin: quantity, optional
        Lower wavelength bound for the spectral filter.
        See :meth:`spectral_filter`.
        If both ``wmin`` and ``wmax`` are ``None``, disable the spectral filter.

    wmax: quantity, optional
        Upper wavelength bound for the spectral filter.
        See :meth:`spectral_filter`.
        If both ``wmin`` and ``wmax`` are ``None``, disable the spectral filter.

    percentage: float, optional
        Percentage value for the integral filter.
        See :meth:`integral_filter`.
        If ``None``, disable the integral filter.

    Notes
    -----
    Select the filtering algorithms corresponding to the specified parameters.

    If multiple filters are selected, they are applied in this order:

    * integral filter
    * spectral filter
    * threshold filter

    See Also
    --------
    :meth:`integral_filter`, :meth:`spectral_filter`,
    :meth:`threshold_filter`
    """
    # convert inputs
    srf = converters.load_dataset(srf)
    output_path = Path(path).absolute()

    # trimming
    trimmed = trim(srf=srf) if trim_prior else srf

    filtered = trimmed

    # integral filter
    if percentage is not None:
        filtered = integral_filter(srf=filtered, percentage=percentage)

    # spectral filter
    if wmin is not None or wmax is not None:
        filtered = spectral_filter(srf=filtered, wmin=wmin, wmax=wmax)

    # threshold filter
    if threshold is not None:
        filtered = threshold_filter(srf=filtered, value=threshold)

    # print filtering summary table
    if verbose:
        table = summarize(srf=srf, filtered=filtered)
        rich.print(table)

    # save filtered dataset
    if interactive:
        show(
            ds=trimmed,
            title=" ".join(
                [srf.attrs["platform"], srf.attrs["instrument"], srf.attrs["band"]]
            ),
            trim_prior=trim_prior,
            threshold=threshold,
            wmin=wmin,
            wmax=wmax,
            percentage=percentage,
        )

        if Confirm.ask("Save filtered dataset?"):
            save(
                ds=filtered,
                path=output_path,
                verbose=verbose,
                dry_run=dry_run,
            )
        else:
            if verbose:
                rich.print("Aborted!")
    else:
        save(
            ds=filtered,
            path=output_path,
            verbose=verbose,
            dry_run=dry_run,
        )
