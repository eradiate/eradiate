"""
Heterogeneous atmospheres
=========================
"""

# %%
# .. warning:: Outdated content, update required

# %%
# This tutorial illustrates how to perform monochromatic simulations in
# one-dimensional heterogeneous atmospheres using the one-dimensional solver
# application :class:`.OneDimSolverApp`. Refer to
# :ref:`sphx_glr_examples_generated_tutorials_solver_onedim_01_solver_onedim.py`
# if needed.

from eradiate.solvers.onedim import OneDimSolverApp

# %%
# Get started
# -----------
#
# In order to focus on the parameters related to the heterogeneous atmosphere,
# we do not specify the surface and illumination configuration sections.
# The :class:`.OneDimScene` created by :class:`.OneDimSolverApp` will use
# defaults to define these elements:
#
# * The default surface is a :class:`.LambertianSurface` with a reflectance
#   value of 0.5.
# * The default illumination is a :class:`.DirectionalIllumination` (0 degree
#   zenith and azimuth angles) with an irradiance value corresponding to the
#   solar irradiance spectrum (as represented by the ``thuillier_2003``
#   dataset).
#
# For the measures section, we will use a ``toa_pplane`` measure because in the
# following we will produce plots of the top-of-atmosphere bidirectional
# reflectance factor in the principal plane.
#
# There is no default for the mode. Therefore, we explicitly set it to
# ``"mono_double"`` (for monochromatic, double precision) and set arbitrarily the
# wavelength to 579 nanometers.

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 579.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "toa_pplane"
    }]
}
app = OneDimSolverApp.from_dict(config)

# %%
# Executing this cell may take a while (typically from 1 to 10 seconds). We tell
# the solver application to create a **heterogeneous atmosphere** with
# ``"type": "heterogeneous"``. Heterogeneous atmospheres can be specified in two
# different ways:
#
# * by setting a so-called ``profile`` attribute
# * by providing paths to extinction coefficient and albedo volume data files.
#
# The second option is for advanced users who have some extinction coefficient
# and albedo volume data files that they would like to use specifically.
# In this tutorial, we will use the first option.
# In heterogeneous atmosphere ``profile`` attribute expects a
# :class:`.RadProfile` object.
# So far, only two types of :class:`.RadProfile` objects are available:
#
# * :class:`.ArrayRadProfile` (``"type": "array"``)
# * :class:`.US76ApproxRadProfile` (``"type": "us76_approx"``)
#
# The :class:`.ArrayRadProfile` is for advanced users who know the values of the
# extinction coefficient and albedo that they would like to use to define the
# radiative properties profile.
# In this tutorial, we will use the :class:`.US76ApproxRadProfile` class to
# define the radiative properties profile.
# This class computes an approximation of the radiative properties profile that
# corresponds to the US76 atmosphere.
# The approximation originates in the absorption coefficient computation, where
# a few simplifications are made (refer to ... for more information).
# The `US76ApproxRadProfile` class takes other parameters but we will use their
# default values for now.
# The solver application is configured and can run the radiative transfer
# simulation:

app.run()

# %%
# Let's have a look at the result in the principal plane:

import matplotlib.pyplot as plt
def visualise(results, ylim=None):
    ds = results['toa_pplane']
    pplane_data = ds.brf
    pplane_data.plot()
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

visualise(app.results, ylim=[0., .6])

# %%
# The results are a little bit too noisy so let us try to get smoother results
# by increasing the number of samples per pixel (the default value was
# ``spp = 32``):

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 579.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "toa_pplane",
        "spp": 65536
    }]
}
app = OneDimSolverApp.from_dict(config)
app.run()
visualise(app.results, ylim=[0, .6])

# %%
# This is much better!
# We see that the BRF values accumulate around the value 0.5, which corresponds
# to the lambertian surface reflectance value that we have used.
# At high zenith angles, we observe that the scene is less reflective, which
# makes sense since the atmosphere medium gets more optically thick when viewed
# from a higher zenith angle.

# %%
# Atmospheric scattering and absorption
# -------------------------------------
#
# We can investigate the atmospheric radiative properties of the heterogeneous
# atmosphere we have just created:

app.scene.atmosphere.profile.sigma_t

# %%
app.scene.atmosphere.profile.albedo

# %%
# There are as many values in the ``sigma_t`` and ``albedo`` arrays that they
# are layers (50 in our example) in the atmosphere.
# The first element corresponds to the first layer (the surface layer), the last
# element corresponds to the last layer (top-of-atmosphere layer).
# The extinction coefficient (``sigma_t``) barely reaches
# :math:`10^{-5} \mathrm{m}^{-1}` at the surface and then decreases by 6 orders
# of magnitude when moving to the top of atmosphere.
# This corresponds to an optically thin participating medium.
# The albedo values are very close to 1, which means that the participating
# medium is almost scattering-only.
# Molecular absorption is very weak in the visible range:

app.scene.atmosphere.profile.sigma_a

# %%
# Molecular absorption gets stronger in the infrared range:

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 1281.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "toa_pplane",
        "spp": 65536
    }]
}
app_1281 = OneDimSolverApp.from_dict(config)
app_1281.scene.atmosphere.profile.albedo

# %%
# We can observe that the TOA BRF values are affected by this more absorbing
# atmosphere:

app_1281.run()

plt.figure()
app_1281.results['toa_pplane'].brf.plot()
plt.plot(
    app.results['toa_pplane'].vza.values,
    app.results['toa_pplane'].brf.values,
    linestyle="dashed",
    color="green"
)
plt.legend(["$\lambda=1281$ nm", "$\lambda=579$ nm"], loc='lower center')
plt.ylim([0., 0.6])
plt.show()

# %%
# Molecular absorption spectra feature so-called "absorption lines": absorption
# occurs in some specific narrow wavelength values ranges.
# Outside of these ranges, there is no absorption.
# Let us change the wavelength from 1281 to 1279 nm:

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 1279.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    },
    "measures": [{
        "type": "toa_pplane",
        "spp": 65536
    }]
}
app = OneDimSolverApp.from_dict(config)
app.scene.atmosphere.profile.albedo

# %%
# Molecular scattering is computed using the Rayleigh scattering model.
# In this model, the scattering coefficient varies proportionaly to
# :math:`\lambda^{-4}` where :math:`\lambda` is the wavelength.
# If we change the wavelength from 580 to 400 nm, we observe that the values of
# the scattering coefficient are larger:

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 400.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx"
        }
    }
}
app = OneDimSolverApp.from_dict(config)
app.scene.atmosphere.profile.sigma_s.to("1/m")

# %%
# Set the atmosphere height and number of layers
# ----------------------------------------------
#
# Next, let's try to customise the heterogeneous atmosphere by changing its
# height and number of layers.
#
# So far, the heterogeneous atmosphere was defined by the ``us76_approx``
# radiative properties profile without parameters.
# The ``us76_approx`` radiative properties profile takes two parameters:
#
# * ``height`` (default: 100 km)
# * ``n_layers`` (default: 50).
#
# Obviously enough, ``height`` sets the height of the radiative properties
# profile, therefore of the heterogeneous atmosphere, whereas ``n_layers`` sets
# the number of atmospheric layers.
#
# Let us try and use different values:

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 580.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx",
            "height": 90,
            "height_units": "km",
            "n_layers": 90
        }
    },
    "measures": [{
        "type": "toa_pplane",
        "spp": 16384
    }]
}
app = OneDimSolverApp.from_dict(config)

# %%
# This creates a 90 km high heterogeneous atmosphere with 90 atmospheric
# layers, each 1 km thick.

app.run()
visualise(app.results, ylim=([0., 0.6]))

# %%
# A 20 km high heterogeneous atmosphere with 10 layers each 2 km thick, would
# be set with:

config = {
    "mode": {
        "type": "mono_double",
        "wavelength": 580.
    },
    "atmosphere": {
        "type": "heterogeneous",
        "profile": {
            "type": "us76_approx",
            "height": 20,
            "height_units": "km",
            "n_layers": 10
        }
    },
    "measures": [{
        "type": "toa_pplane",
        "spp": 65536
    }]
}
app = OneDimSolverApp.from_dict(config)
app.run()
visualise(app.results, ylim=([0., 0.6]))
