.. _sec-atmosphere-intro:

Introduction
============

Assumptions
-----------

The air is assumed to behave as an
`ideal gas <https://en.wikipedia.org/wiki/Ideal_gas>`_,
i.e. a collection of point particles without interactions between one another.
It is in thermodynamic equilibrium, it is not chemically reacting and it obeys
the ideal gas state equation:

.. math::

   p = n k T

where:

* :math:`p` stands for the pressure :math:`[ML^{-1}T^{-2}]`,
* :math:`n` stands for the number density :math:`[L^{-3}]`,
* :math:`k` is the
  `Bolzmann constant <https://en.wikipedia.org/wiki/Boltzmann_constant>`_
  :math:`[ML^{2}T^{-2}\Theta^{-1}]` and
* :math:`T` stands for the temperature :math:`[\Theta]`.

Air can usually be treated as an ideal gas within reasonable tolerance over a
wide parameter range around standard temperature and pressure (273 K, 100 kPa)
and the approximation generally gets better with lower pressure and higher
temperature.

Atmosphere modelling
--------------------

Eradiate represents the atmosphere as a **shape** that encompasses a
**participating medium** (the air).

The shape is a cuboid characterised by width and height values that correspond
to the width and height, respectively, of the atmosphere object.

.. image:: ../../../fig/atmosphere-cuboid_shape.png
   :align: center
   :scale: 50

This shape delimits the region of space that is occupied by the participating
medium.

The participating medium is characterised by a phase function type and two 3D
arrays of albedo and extinction coefficient values that describe how
radiative properties vary in space.

.. image:: ../../../fig/atmosphere-participating_medium.png
   :align: center
   :scale: 50

Each value in these 3D array corresponds to one cell of a spatial mesh that
discretises the participating medium into an arrangement of adjacent 3D cells
wherein the radiative properties are uniform.
In the example illustrated by the image above, the shape of these arrays would
be (4, 2, 2).
The extinction coefficient (:math:`k_{\mathrm{t}}`) and albedo (:math:`\varpi`)
are defined by:

.. math::

   k_{\mathrm{t}} = k_{\mathrm{s}} + k_{\mathrm{a}}

.. math::

   \varpi = \frac{k_{\mathrm{s}}}{k_{\mathrm{t}}}

where:

* :math:`k_{\mathrm{a}}` is the absorption coefficient :math:`[L^{-1}]` and
* :math:`k_{\mathrm{s}}` is the scattering coefficient :math:`[L^{-1}]`.

The phase function does not vary from one cell to the other ; it is the same
for the whole atmosphere.
Only the albedo and extinction coefficient are allowed to vary with space.

The structure of the participating medium is assumed to be isotropic, i.e. the
properties of the medium embedded in one cell is invariant to rotation of the
cell.

.. note::
   So far, only purely molecular atmospheres are supported.
   Work on adding aerosols to the atmosphere is ongoing.

Atmosphere types
----------------

Eradiate provides two atmosphere types:

* homogeneous atmosphere
  (:class:`~eradiate.scenes.atmosphere.HomogeneousAtmosphere`): radiative
  properties are uniform within the atmosphere.
* heterogeneous atmosphere
  (:class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere`): radiative
  properties are non-uniform within the atmosphere.

.. image:: ../../../fig/atmosphere-classes.png
   :align: center

Both atmosphere types inherit an abstract atmosphere base type
(:class:`~eradiate.scenes.atmosphere.Atmosphere`),
parameterised by a top-of-atmosphere altitude (``toa_altitude``) and a width
(``width``).
As a result, these parameters can be set for both atmosphere types.
Both atmosphere types use the
:ref:`Rayleigh scattering phase function <sec-atmosphere-molecular-scattering>`
to describe the angular distribution of scattered light.

Homogeneous atmosphere
~~~~~~~~~~~~~~~~~~~~~~

The homogeneous atmosphere is characterised by uniform extinction coefficient
(:math:`k_{\mathrm{s}}`) and albedo (:math:`\varpi`).

.. image:: ../../../fig/atmosphere-homogeneous.png
   :align: center
   :scale: 50

.. admonition:: Example

   Create a 1000 x 1000 x 120 km homogeneous atmosphere with
   :math:`k_{\mathrm{s}} = 10^{-3} \, \mathrm{km}^{-1}` and
   :math:`k_{\mathrm{a}} = 10^{-5} \, \mathrm{km}^{-1}`
   using:

   .. code:: python

      import eradiate
      from eradiate import unit_registry as ureg
      eradiate.set_mode("mono")

      atmosphere = eradiate.scenes.atmosphere.HomogeneousAtmosphere(
          toa_altitude = ureg.Quantity(120, "km"),
          width = ureg.Quantity(1000, "km"),
          sigma_s = ureg.Quantity(1e-3, "km^-1"),
          sigma_a = ureg.Quantity(1e-5, "km^-1"),
      )

For more detail, refer to the guide page dedicated to
:ref:`homogeneous atmospheres <sec-atmosphere-homogeneous>`.

Heterogeneous atmosphere
~~~~~~~~~~~~~~~~~~~~~~~~

The heterogeneous atmosphere is characterised by non-uniform radiative
properties.

.. note::

   At the moment, Eradiate provides 1D-heterogeneous atmosphere only.

.. image:: ../../../fig/atmosphere-heterogeneous.png
   :align: center
   :scale: 50

1D-heterogeneous atmospheres are characterised by radiative properties
that vary with altitude.
Together, the values of the radiative properties at different altitude points
constitute the **radiative properties profile** of the atmosphere.
The radiative properties profile is specified by the ``profile`` parameter of
the :class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere` class.
You can either specify the radiative properties profile by choosing
a profile from a list of registered profile types, or by providing
:ref:`kernel volume data files <sec-atmosphere-heterogeneous-kernel_volume_data_files>`
where the values of the extinction coefficient and albedo are written.

.. note::

   So far, only the
   :class:`us76_approx <eradiate.radprops.rad_profile.US76ApproxRadProfile>`
   radiative properties profile is available.
   We are working on adding radiative properties profiles corresponding to the
   AFGL (1986) atmospheric profiles
   :cite:`Anderson1986AtmosphericConstituentProfiles`.
   Stay tuned for that!

.. admonition:: Example

   Create a 1000 x 1000 x 120 km heterogeneous atmosphere with a radiative
   properties profile approximately corresponding to the US76 atmosphere, using:

   .. code:: python

      import eradiate
      from eradiate import unit_registry as ureg
      eradiate.set_mode("mono")

      atmosphere = eradiate.scenes.atmosphere.HeterogeneousAtmosphere(
          toa_altitude="auto",
          width=ureg.Quantity(1000, "km"),
          profile=dict(
              type="us76_approx",
              height=ureg.Quantity(120, "km"),
              n_layers=50,
          )
      )

   In this example, the atmosphere extends from 0 km to 120 km and is divided
   into 50 layers.
   In each of these layers, the albedo and the extinction coefficient are
   automatically computed in the appropriate pressure and temperature conditions
   corresponding to the US76 atmosphere, and at the current wavelength.

For more detail, refer to the guide page dedicated to
:ref:`heterogeneous atmospheres <sec-atmosphere-heterogeneous>`.
