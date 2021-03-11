.. _sec-atmosphere-heterogeneous:

Heterogeneous atmospheres
=========================

The non-uniform radiative properties of the heterogeneous atmosphere are
described by the radiative properties profile, which consists of the two
spatial fields
:math:`\varpi (x,y,z)` and
:math:`k_{\mathrm{t}} (x,y,z)`
where:

* :math:`\varpi` is the albedo :math:`[/]` and
* :math:`k_{\mathrm{t}}` is the extinction coefficient :math:`[L^{-1}]`.

.. note::

   So far, only 1D-heterogeneous atmospheres are available.
   The radiative properties profile thus consists of the spatial fields
   :math:`\varpi (z)` and
   :math:`k_{\mathrm{t}} (z)`

The :class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere` constructor
requires that you provide the radiative properties profile using either

* a :class:`~eradiate.radprops.rad_profile.RadProfile` object or
* kernel volume data files.

Radiative properties profile using a ``RadProfile`` object
----------------------------------------------------------

There are several types of ``RadProfile`` objects available.
List them with:

.. code:: python

   import eradiate

   list(eradiate.radprops.RadProfileFactory.registry)

us76_approx
~~~~~~~~~~~

To create a heterogeneous atmosphere based on the
:class:`us76_approx <eradiate.radprops.rad_profile.US76ApproxRadProfile>`
radiative properties profile type, use:

.. code:: python

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       profile={
           "type": "us76_approx",
       },
   )

The default ``us76_approx`` radiative properties profile extends to 100 km
in height and includes 50 layers (each 2 km thick).
You can customise the radiative properties profile.
Set the ``height`` parameter to determine the vertical extent of
the corresponding atmosphere, and the ``n_layers`` parameters to specify the
number of layers the atmosphere is divided into:

.. code::

   from eradiate import unit_context_config as ucc
   ucc.override({"length": "km"}):

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       profile={
           "type": "us76_approx",
           "height": 120.,
           "n_layers": 120,
       },
   )

The corresponding atmosphere is 120 km high and is divided into 120 layers
(each 1 km thick).

.. note::

   The layers are of equal thickness.

.. note::

   When you set the ``height`` parameter of the ``us76_approx`` profile object,
   the corresponding atmosphere ``toa_altitude`` is automatically adjusted to
   that value.
   In fact, you cannot set both the ``profile`` and ``toa_altitude`` attributes
   of the :class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere` class.
   The following code, for example, would raise an exception:

   .. code:: python

      atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
          toa_altitude=100,
          profile={
              "type": "us76_approx",
              "height": 120.,
              "n_layers": 120,
          },
      )

   For more detail, refer to the
   :class:`~eradiate.radprops.rad_profile.US76ApproxRadProfile`
   reference documentation.

By default, the width of the heterogeneous atmosphere is set to 1000 km.
This width guarantees the absence of edge effects in simulations where the
sensor is a radiance meter array placed at the top of the atmosphere and looking
down with a zenith angle varying from 0 to 75°.
Above 75°, the measured values start to be influenced by the fact that the
horizontal size of the atmosphere is finite.
For accurate results above 75°, consider increasing the atmosphere width, using
the ``width`` attribute:

.. code:: python

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       width=1e4,
       profile={
           "type": "us76_approx",
           "height": 120.,
           "n_layers": 120,
       },
   )

array
~~~~~

To create a heterogeneous atmosphere based on the
:class:`array <eradiate.radprops.rad_profile.ArrayRadProfile>` radiative
properties profile type, use:

.. code:: python

   import numpy as np

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       profile={
           "type": "array",
           "sigma_t_values": np.array([1e-5, 1e-6, 1e-7, 1e-8]).reshape(1, 1, 4),
           "albedo_values": np.array([.95, .97, .99, 1.]).reshape(1, 1, 4),
           "height": 100.,
       },
   )

The corresponding atmosphere is 100 km high and is divided into 4 layers
(each 25 km thick).
The first values in the  ``sigma_t_values`` and ``albedo_values`` arrays
correspond to the bottom layer of the atmosphere.

Kernel volume data files
------------------------

When the heterogeneous atmosphere object is created, the radiative properties
are written to files, which can be accessed afterwards.
The locations of these data files is stored in the ``albedo_fname`` and
``sigma_t_fname`` attributes.
By default, these files are placed in a temporary directory with a random name.
To control where these files are saved, set the ``albedo_fname`` and
``sigma_t_fname`` attributes:

.. code:: python

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       albedo_fname="albedo.vol",
       sigma_t_fname="sigma_t.vol",
       profile={
           "type": "us76_approx",
           "height": 120.,
           "n_layers": 120,
       },
   )

Later, you can re-use these files to create the same heterogeneous atmosphere:

.. code:: python

   atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
       toa_altitude=120.,
       albedo_fname="albedo.vol",
       sigma_t_fname="sigma_t.vol",
   )

.. note::

   You must set again the top-of-atmosphere altitude, because the kernel
   volume data files only hold the radiative properties.
