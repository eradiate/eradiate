import drjit as dr
import mitsuba as mi
import numpy as np
import xarray as xr


def _sph_to_dir(theta, phi):
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)


def _eval_bsdf_impl(plugin, theta_os, phi_os, theta_i, phi_i):
    theta_ov, phi_ov = dr.meshgrid(theta_os, phi_os)
    wos = _sph_to_dir(theta_ov, phi_ov)

    # Evaluate BSDF
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = _sph_to_dir(theta_i, phi_i)
    values = plugin.eval(mi.BSDFContext(), si, wos)
    return np.reshape(values, (len(phi_os), len(theta_os))).T


def eval_bsdf(plugin, theta_os, phi_os, theta_is, phi_is) -> xr.Dataset:
    # Requires a JIT variant
    if mi.Float == mi.ScalarFloat:
        raise RuntimeError("A JIT-compiled variant is required")

    result = [
        [
            _eval_bsdf_impl(plugin, theta_os, phi_os, theta_i, phi_i)
            for theta_i in theta_is
        ]
        for phi_i in phi_is
    ]

    return xr.Dataset(
        {
            "bsdf": (
                ("theta_o", "phi_o", "theta_i", "phi_i"),
                np.transpose(result, (2, 3, 1, 0)),
                {"units": "sr^-1"},
            )
        },
        coords={
            "theta_o": ("theta_o", theta_os, {"units": "rad"}),
            "phi_o": ("phi_o", phi_os, {"units": "rad"}),
            "theta_i": ("theta_i", theta_is, {"units": "rad"}),
            "phi_i": ("phi_i", phi_is, {"units": "rad"}),
        },
    )
