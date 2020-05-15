Scene generation API reference
==============================

Fundamental classes
-------------------

.. currentmodule:: eradiate.scenes.builder.base

.. autosummary::
    :toctree: ../stubs/base
    :template: autoclass.rst
    :nosignatures:

    Object
    Instantiable
    Plugin
    ReferablePlugin
    Ref
    Bool
    Int
    Float
    String
    Point
    Vector

BSDFs
-----

.. currentmodule:: eradiate.scenes.builder.bsdfs

.. autosummary::
    :toctree: ../stubs/bsdfs
    :template: autoclass.rst
    :nosignatures:

    BSDF
    Null
    Diffuse
    RoughDielectric

Emitters
--------

.. currentmodule:: eradiate.scenes.builder.emitters

.. autosummary::
    :toctree: ../stubs/emitters
    :template: autoclass.rst
    :nosignatures:

    Emitter
    Constant
    Directional
    Area

Films
-----

.. currentmodule:: eradiate.scenes.builder.films

.. autosummary::
    :toctree: ../stubs/films
    :template: autoclass.rst
    :nosignatures:

    Film
    HDRFilm

Integrators
-----------

.. currentmodule:: eradiate.scenes.builder.integrators

.. autosummary::
    :toctree: ../stubs/integrators
    :template: autoclass.rst
    :nosignatures:

    Integrator
    Direct
    Path
    VolPath

Media
-----

.. currentmodule:: eradiate.scenes.builder.media

.. autosummary::
    :toctree: ../stubs/media
    :template: autoclass.rst
    :nosignatures:

    Medium
    Homogeneous

Phase functions
---------------

.. currentmodule:: eradiate.scenes.builder.phase

.. autosummary::
    :toctree: ../stubs/phase
    :template: autoclass.rst
    :nosignatures:

    PhaseFunction
    Isotropic
    HenyeyGreenstein
    Rayleigh

Reconstruction filters
----------------------

.. currentmodule:: eradiate.scenes.builder.rfilters

.. autosummary::
    :toctree: ../stubs/rfilters
    :template: autoclass.rst
    :nosignatures:

    ReconstructionFilter
    Box

Samplers
--------

.. currentmodule:: eradiate.scenes.builder.samplers

.. autosummary::
    :toctree: ../stubs/samplers
    :template: autoclass.rst
    :nosignatures:

    Sampler
    Independent

Scene
-----

.. currentmodule:: eradiate.scenes.builder.scene

.. autosummary::
    :toctree: ../stubs/scene
    :template: autoclass.rst
    :nosignatures:

    Scene

Sensors
-------

.. currentmodule:: eradiate.scenes.builder.sensors

.. autosummary::
    :toctree: ../stubs/sensors
    :template: autoclass.rst
    :nosignatures:

    Sensor
    Perspective
    RadianceMeter
    Distant

Shapes
------

.. currentmodule:: eradiate.scenes.builder.shapes

.. autosummary::
    :toctree: ../stubs/sensors
    :template: autoclass.rst
    :nosignatures:

    Shape
    Rectangle
    Cube

Spectra
-------

.. currentmodule:: eradiate.scenes.builder.spectra

.. autosummary::
    :toctree: ../stubs/sensors
    :template: autoclass.rst
    :nosignatures:

    Spectrum

Textures
--------

.. currentmodule:: eradiate.scenes.builder.textures

.. autosummary::
    :toctree: ../stubs/textures
    :template: autoclass.rst
    :nosignatures:

    Texture
    Checkerboard

Transforms
----------

.. currentmodule:: eradiate.scenes.builder.transforms

.. autosummary::
    :toctree: ../stubs/transforms
    :template: autoclass.rst
    :nosignatures:

    Transform
    LookAt
    Rotate
    Scale
    Translate