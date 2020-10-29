.. _sec-onedim_application-introduction:

One dimensional solver application (``OneDimSolverApp``)
====================================================

Intro
-----

[This is the hardest part to write, I feel]

Available features
------------------

.. admonition:: Stuff that should go here

    - Lambertian Surface
    - RPV Surface [should we mention our reference for implementation here?]
    - Directional Illumination
    - Constant Illumination
    - Homogeneous rayleigh scattering only atmosphere
    - Heterogeneous atmosphere with absorption
    - Top of atmosphere measure hemisphere and principal plane
    - Outgoing radiance, BRDF, BRF

OneDimSolverApp offers a range of features that allow for the computation of typical quantities in atmospheric
radiation transfer.

Illumination
^^^^^^^^^^^^

As of the current version two types of emitters are available in OneDimSolverApp. The **constant** source
illuminates the scenes from all directions homogeneously, while the **directional** source implements an
infinitely distant emitter, which emits light into a single direction.

Measure
^^^^^^^

OneDimSolverApp contains features to compute the top of **atmosphere leaving radiance**, either into the entire **hemisphere**
or restricted to only the **principal plane**. The latter can reduce computational time if a user is only interested in 
the principal plane by not computing the leaving radiance for directions outside of the principal plane.

Atmosphere
^^^^^^^^^^

Atmosphere specification is split in two general types. **Homogeneous**  atmospheres exhibit spatially invariant radiative properties
and does not include absorption of radiation. This type of atmosphere only implements rayleigh scattering.
On the other hand **heterogeneous** atmospheres include absorption of radiation as well as elevation dependent radiative properties.

Surfaces
^^^^^^^^

Naturally one dimensional problems include only limited surface features. Accordingly OneDimSolverApp includes the most
common spatially invariant surface models. The **lambertian** model exhibits isotropic scattering, irrespective of the incident light geometry.
The **RPV** model implements the variants with three and four parameters.


Configuring the application
---------------------------

Following the design principles of Eradiate, OneDimSolverApp can be configured in two ways.
In interactive python sessions such as ipython or jupyter, the configuration is written into a
python dictionary. When using the application through its command line interface, configuration
is parsed from a yaml file.

This section aims at introducing the basics of configuring the application through both mechanisms.

An example configuration in yaml might look like this:

.. code-block:: yaml

    mode:
        type: mono
        wavelength: 577.
    surface:
        type: rpv
    atmosphere:
        type: rayleigh_homogeneous
        height: 40
        height_units: km
        sigma_s: 1.e-4
    illumination:
        type: directional
        zenith: 30.
        azimuth: 0.
        irradiance: 
            type: uniform
            value: 1.8e+6
            value_unit: W/km**2/nm
    measure:
        - type: toa_lo_hsphere
            spp: 32000
            zenith_res: 5.
            azimuth_res: 5.

The corresponding dictionary in an interactive python session would look like this:

.. code-block:: python

    {
        "mode": {
            "type": "mono",
            "wavelength": 577.
        },
        "surface": {
            "type": "rpv"
        },
        "atmosphere": {
            "type": "rayleigh_homogeneous",
            "height": 40,
            "height_units": "km",
            "sigma_s": 1.e-4
        },
        "illumination": {
            "type": "directional",
            "zenith": 30.,
            "azimuth": 0.,
            "irradiance": {
                "type": "uniform",
                "value": 1.8e+6,
                "value_units": "W/km**2/nm"
            },
        },
        "measure": [{
            "type": "toa_lo_hsphere",
            "spp": 32000,
            "zenith_res": 5,
            "azimut_res": 5
        }]
    }

In the following, the sections with their options will be discussed. For a thorough discussion
of all options and their parameters, please refer to the :ref:`sec-api_reference-intro`.

Mode
^^^^

The mode section sets internal details of Eradiate's computational kernel. DoneDimSolverApp 
only supports one mode, called mono. In this mode a monochromatic simulation is performed, 
which means that only one wavelength is transported per ray. Additionally the mono mode sets 
the wavelength for the simulation which is used to compute the scattering parameters of the atmosphere.

Surface
^^^^^^^

Options for surface specification are limited, since one dimensional computations can only employ spatially
invariant models for surface scattering. The available variants are lambertian scattering and the RPV
scattering model with three or four parameters.

In the example above a RPV atmosphere with default parameters is selected.

Atmosphere
^^^^^^^^^^

Eradiate currently offers two kinds of atmospheric models. The homogeneous model provides a spatially invariant
medium, which expresses only rayleigh scattering with a parametrizable scattering cross section.
The heterogeneous model on the other hand provides a model with height dependent scattering properties
which includes scattering and absorption.

The heterogeneous atmosphere model combines atmospheric profiles, such as US76 [add a reference here] with 
empirical absorption spectra for a variety of gas species.

In this example, the atmosphere is set to a homogeneous variant, with a height of 40 kilometers and a scattering
cross section of 1e-4 1/m.

Illumination
^^^^^^^^^^^^

The illumination section defines the light source in the simulation. Currently two illumination variants
are available, directional illumination, which implements light from an infinitely distant emitter that
emits into a signel direction and constant illumination, which illuminates the scene homogeneously from
all directions.

Here a directional light source is used. This plugin's parameters set the emitter's direction through 
its zenith and azimuth angle (also known as sun zenith angle and sun azimuth angle) and its irradiance 
given in units of Watts per square kilometer per nanometer.


Measure
^^^^^^^

The measure section defines the observational parameters.
Currently only the top of atmosphere leaving radiance into the entire hemisphere or into the principal planecan be measured.
The BRDF and BRF are computed in post processing

Here a hemispherical measure is chosen, which means that the leaving radiance into the entire hemisphere will be recorded. 
The spp parameter defines the number of samples drawn per observational direction and zenith_res and azimuth_res 
define the angular resolution in zenith and azimuth direction in degrees respectively.

Note that the Bidirectional reflectance distribution function (BRDF) and bidirectional reflectance factor (BRF) are automatically computed from the leaving radiance 
in post processing.

Data output and visualisation
-----------------------------

Result output
^^^^^^^^^^^^^

Data output depends on the way, the application is accessed. Using the CLI, users can specify a positional argument 
``fname_results`` which will be used as a file name prefix for all results output. Data is written to a NetCDF file
in version 4 or 3 of the format, depending on the available libraries on the executing machine.

In an interactive session the ``run()`` command returns a list of :class:`~xarray.Dataset` objects.
The list will contain one entry for each measure, defined in the configuration file. Each of the datasets will
contain one data array for the leaving radiance, brdf and brf for the corresponding measure.

Plotting
^^^^^^^^

Additionally OneDimSolverApp provides basic plotting facilities, which give an overview over the computed results.
In the CLI the parameter ``fname_plots`` will enable plotting and create a set of plots, depicting 
all computed quantities. Hemispherical quantities will be represented with polar plots, while principal plane
measures will be represented using linear plots.

Interactive use of OneDimSolverApp will not automatically output plots but let the user decide which quantities
to output.

[should we put a code example here or is this explained elsewhere?]
