"""
Discrete canopy tutorial
========================
"""

# %%
# This tutorial provides practical examples to use the :class:`.DiscreteCanopy`
# class. As usual, we start by importing the Eradiate module and setting up
# an alias to the unit registry:

import eradiate

eradiate.set_mode("mono")
ureg = eradiate.unit_registry

# %%
# We then define a small utility function leveraging the low-level components
# of Eradiate to visualise conveniently the canopies we will create throughout
# this tutorial:

import matplotlib.pyplot as plt
import numpy as np


def display_canopy(canopy, distance=85):
    # Compute camera location
    origin = np.full((3,), distance / np.sqrt(3))

    # Prepare kernel evaluation context
    ctx = eradiate.contexts.KernelDictContext()

    # Build kernel scene dictionary suitable for visualisation
    # (we'll render only direct illumination for optimal speed)
    kernel_dict = eradiate.scenes.core.KernelDict.new(
        eradiate.scenes.measure.PerspectiveCameraMeasure(
            id="camera",
            film_resolution=(640, 480),
            spp=32,
            origin=origin,
            target=(0, 0, 0),
            up=(0, 0, 1),
            spectral_cfg=ctx.spectral_ctx,
        ),
        eradiate.scenes.illumination.DirectionalIllumination(),
        eradiate.scenes.integrators.PathIntegrator(max_depth=2),
        canopy,
        ctx=ctx,
    )

    # Render image
    scene = kernel_dict.load()
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)

    # Display image
    img = np.array(sensor.film().bitmap()).squeeze()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img, cmap=plt.get_cmap("Greys_r"))
    plt.colorbar(im, fraction=0.0354, pad=0.02)
    return ax


# %%
# Eradiate ships a powerful interface to generate *abstract discrete canopies*
# consisting in a set of disc-shaped "leaves" with a bilambertian surface
# scattering model. This tutorial introduces basic and advanced usage of this
# interface.

# %%
# Basic interface principles
# --------------------------
#
# Abstract discrete canopies are implemented by the :class:`.DiscreteCanopy`
# class. It is available as a top-level scene element, meaning that it can be
# directly added to a scene and instantiated using the
# :class:`.BiosphereFactory`.
#
# A :class:`.DiscreteCanopy` consists of one or several
# :class:`.CanopyElement`s, *i.e.* set of floating discs and possible trunks.
# :class:`.CanopyElement` implements two sub-classes, :class:`.LeafCloud` and
# :class:`.AbstractTree`, the latter holds a :class:`.LeafCloud` plus
# specification for a cylindrical trunk. The leaf clouds can be made of a large
# number of discs; if the requested :class:`.DiscreteCanopy` object contains
# too many of them (say, several millions), the memory footprint of the
# resulting scene will quickly become large.
#
# In many situations, however, we can simply define a small number of leaf
# clouds and clone them at designated locations. Under the hood,
# :class:`.DiscreteCanopy` does not reference directly a set of
# :class:`.CanopyElement` objects; instead, they are wrapped into
# :class:`.InstancedCanopyElement` instances which associate to a leaf cloud
# or abstract tree the positions at which it should be *instanced*
# (*i.e.* cloned).

# %%
# Quickly create a homogeneous discrete canopy
# --------------------------------------------
#
# If you are in a hurry and just want to quickly add a homogeneous discrete
# canopy to your scene, the :meth:`.DiscreteCanopy.homogeneous` class
# method constructor forwards directly its parameters to the
# :meth:`.LeafCloud.cuboid` class method constructor. The following
# call will generate a homogeneous discrete canopy of 10 m x 10 m x 3 m size
# with leaves of 10 cm radius and leaf area index equal to 3:

canopy = eradiate.scenes.biosphere.DiscreteCanopy.homogeneous(
    lai=3, leaf_radius=10.0 * ureg.cm, l_horizontal=10 * ureg.m, l_vertical=3 * ureg.m
)
display_canopy(canopy, distance=50)
plt.show()

# %%
# These parameters, or a set of dependent parameters from which they can be
# computed, are required (see :class:`.CuboidLeafCloudParams`). In addition, we
# can specify leaf optical properties:

eradiate.scenes.biosphere.DiscreteCanopy.homogeneous(
    lai=3,
    leaf_radius=10 * ureg.cm,
    l_horizontal=10 * ureg.m,
    l_vertical=3 * ureg.m,
    leaf_transmittance=0.2,
    leaf_reflectance=0.4,
)

# %%
# We can use a factory to make an equivalent call. For that purpose, we
# will pack the keyword arguments into a dictionary. In addition, we need to
# instruct the :meth:`.factory.convert` method to dispatch our
# dictionary parameters to :meth:`.DiscreteCanopy.homogeneous` using
# the ``construct`` parameter:

eradiate.scenes.biosphere.biosphere_factory.convert(
    {
        "type": "discrete_canopy",
        "construct": "homogeneous",
        "n_leaves": 1,
        "leaf_radius": 0.1,
        "l_horizontal": 10,
        "l_vertical": 3,
    }
)

# %%
# Load canopy specifications from files
# -------------------------------------
#
# :class:`.DiscreteCanopy` and the classes it encapsulates provide a flexible
# interface to load discrete canopies specifications from files. Leaf clouds can
# be completely specified using the :meth:`.LeafCloud.from_file` class method
# constructor (see its documentation for the file format). We will load the
# leaf cloud specification used in the
# `"floating spheres" <https://rami-benchmark.jrc.ec.europa.eu/_www/phase_descr.php?strPhase=RAMI3#inline-nav-descr>`_
# RAMI test case series (the leaf cloud specification file can be downloaded
# `here <https://rami-benchmark.jrc.ec.europa.eu/_www/RAMI3/images/HET01_UNI_scene.def>`_).
# The following code snippet creates a single leaf cloud centered at the local
# frame origin:

# We use the path resolver to get the absolute path to the data file
# located in the $ERADIATE_DIR/resources/data/tests/canopies directory
leaf_cloud_filename = eradiate.path_resolver.resolve(
    "tests/canopies/HET01_UNI_scene.def"
)

leaf_cloud = eradiate.scenes.biosphere.LeafCloud.from_file(
    id="floating_spheres_leaf_cloud", filename=leaf_cloud_filename
)
leaf_cloud

# %%
# We can proceed similarly with loading instance positions. The Joint Research
# Center provides no file for us to do it, but we ship one in the Eradiate data
# directory:

# We use the path resolver to get the absolute path to the data file
# located in the $ERADIATE_DIR/resources/data/tests/canopies directory
instance_filename = eradiate.path_resolver.resolve(
    "tests/canopies/HET01_UNI_instances.def"
)

instanced_leaf_cloud = eradiate.scenes.biosphere.InstancedCanopyElement.from_file(
    filename=instance_filename, canopy_element=leaf_cloud
)
instanced_leaf_cloud

# %%
# We can then use these two objects to define a :class:`.DiscreteCanopy`:

canopy = eradiate.scenes.biosphere.DiscreteCanopy(
    id="floating_spheres",
    size=[100, 100, 30] * ureg.m,
    instanced_canopy_elements=instanced_leaf_cloud,
)
canopy

# %%
# Let's render an image of this canopy:

display_canopy(canopy, distance=200)
plt.show()

# %%
# This can be repeated to specify as many leaf clouds and associated instances
# as needed. Since this use case is quite common, the
# :meth:`.DiscreteCanopy.leaf_cloud_from_files` class method constructor
# provides a more convenient interface (it notably takes care by itself of
# setting scene element IDs consistently):


canopy = eradiate.scenes.biosphere.DiscreteCanopy.leaf_cloud_from_files(
    id="floating_spheres",
    size=[100, 100, 30] * ureg.m,
    leaf_cloud_dicts=[
        {
            "instance_filename": instance_filename,
            "leaf_cloud_filename": leaf_cloud_filename,
            "leaf_reflectance": 0.4,
            "leaf_transmittance": 0.1,
        }
    ],
)
canopy

# %%
# Padding
# -------
#
# Practical applications sometimes requires to further "clone" the canopy in order
# to account for adjacency effects. Eradiate allows to pad a canopy with copies
# of itself for that purpose, using the :meth:`.DiscreteCanopy.padded` method:
# it returns a grid of copies of the canopy.

canopy = eradiate.scenes.biosphere.DiscreteCanopy.homogeneous(
    lai=3, leaf_radius=10.0 * ureg.cm, l_horizontal=10 * ureg.m, l_vertical=3 * ureg.m
)
padded_canopy = canopy.padded_copy(1)

print(f"Canopy:")
print(f"  size: {canopy.size}")
print(
    f"  # instances: {canopy.instanced_canopy_elements[0].instance_positions.shape[0]}"
)
display_canopy(canopy, distance=50)
plt.show()

print(f"Padded canopy:")
print(f"  size: {padded_canopy.size}")
print(
    f"  # instances: {padded_canopy.instanced_canopy_elements[0].instance_positions.shape[0]}"
)
display_canopy(padded_canopy, distance=50)
plt.show()
