Data post-processing and visualization
======================================

BRDF Viewer
-----------

The BRDF viewer is a utility for the visualization of BRDF data. In its current
state it handles Eradiate kernel BSDF plugins and gridded data stored in 
`xarray <https://xarray.pydata.org/en/stable/>`_ types.

The viewer offers two modes of visualization, a polar hemispherical plot and a
principal plane plot. This page introduces the utility and guides new users to
its use.

XArray data format specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The XArray adapter can be used to display results from radiative transfer computation
performed with Eradiate. Data is expected to be formatted in the following way:

- The array has 5 dimensions, named and ordered like this:
  :code:`['theta_i', 'phi_i', 'theta_o', 'phi_o', 'wavelength']`
- The phi_o dimension must contain values for at least one of the two values
  0° and 360° for interpolation. If only one is present, it is used for
  interpolation from both directions
- Dimensions along which only one value is provided (e.g. only one wavelength
  was used in the computation) cannot be interpolated and the exact value must
  be chosen to parametrize the view.

Hemispherical view
^^^^^^^^^^^^^^^^^^

The hemispherical view creates a contour plot, depicting light scattering for a
fixed incoming direction into the hemisphere containing the surface normal.
The following code snippet details the creation of a hemispherical view from an
Eradiate kernel BSDF plugin, with the individual steps explained below.

.. code-block:: python

    import eradiate
    import eradiate.util.brdf_viewer as bv
    from eradiate.kernel.core.xml import load_string

    bsdf = {
      'type':  'roughdielectric',
      'distribution': 'beckmann',
      'alpha': 0.1,
      'int_ior': 'bk7',
      'ext_ior': 'air'
    }

    view = bv.PolarView()
    view.zen_res = 1
    view.azm_res = 1
    view.wi = (45,0)
    view.brdf = bsdf

    view.evaluate()
    view.plot()

The second and third import statements are not strictly necessary but provide
useful shorthand notations.

In this example a BSDF of the type :code:`roughdielectric` is created by
parsing an XML snippet through the xml import facilities.

Next, a :code:`PolarView` object is created and assigned the alias :code:`view`.

Setting its properties is very easy:

* The zenith angle resolution is set to one degree, by setting the :code:`zen_res`
  variable
* The azimuth angle resolution is set to one degree, by setting the :code:`zen_res`
  variable
* The direction for the incoming radiation is set to 45 degrees zenith and 0 degrees
  azimuth
* The :code:`PolarView` object is given the BSDF object we created earlier

The :code:`evaluate()` method probes the BSDF object for all directions and fills
the internal data storage. Finally the :code:`plot()` method creates the polar
plot and displays it.

PrincipalPlaneView
^^^^^^^^^^^^^^^^^^

The principal plane view depicts light scattering into the plane that coincides
with the surface normal and the direction of incoming radiation.
The following code snippet details the creation of a principal plane view from 
an Eradiate kernel BSDF plugin, with the individual steps explained below.

.. code-block:: python

    import eradiate
    import eradiate.util.brdf_viewer as bv
    from eradiate.kernel.core.xml import load_string

    bsdf = load_string("""<bsdf version='2.0.0' type="roughdielectric">
        <string name="distribution" value="beckmann"/>
        <float name="alpha" value="0.1"/>
        <string name="int_ior" value="bk7"/>
        <string name="ext_ior" value="air"/>
    </bsdf>""")

    view = bv.PrincipalPlaneView()
    view.zen_res = 1
    view.wi = (45,0)
    view.brdf = bsdf

    view.evaluate()
    view.plot()

The second and third import statements are not strictly necessary but provide
useful shorthand notations.

In this example a BSDF of the type :code:`roughdielectric` is created by
parsing an XML snippet through the xml import facilities.

Next, a :code:`PrincipalPlaneView` object is created and assigned the alias :code:`view`.

Setting its properties is very easy:

* The zenith angle resolution is set to one degree, by setting the :code:`zen_res`
  variable
* The direction for the incoming radiation is set to 45 degrees zenith and 0 degrees
  azimuth
* The :code:`PrincipalPlaneView` object is given the BSDF object we created earlier

The :code:`evaluate()` method probes the BSDF object for all directions and fills
the internal data storage. Finally the :code:`plot()` method creates the polar
plot and displays it.

Note that the azimuth resolution is not set here, since this plot only contains
one azimuth direction.

Option overview
^^^^^^^^^^^^^^^

The BRDFViewer offers some flexibility on how parameters can be set:

* Zenith and azimuth resolutions can be set as resolutions in degrees, using the
  :code:`zen_res` and :code:`azm_res` variables or as a number of steps, using
  the :code:`zen_steps` and :code:`azm_steps` variables
* The direction of incoming light can be set by any iterable object with two or
  three elements. 

  * An object with three elements is interpreted as a cartesian vector
  * An object with two elements is interpreted as a pair of zenith and azimuth
    angles

* The :code:`plot()` method accepts and returns a :code:`Matplotlib.Axes` object
  which can be used in custom plotting setups. If no :code:`Axes` object is
  provided, the script will create a simple plot to display the results.
* Visualization data can be exported to an 
  `xarray <https://xarray.pydata.org/en/stable/>`_ object, for storage,
  sharing and further processing.

.. note::
    Add example plots, once I know how to handle Axes objects