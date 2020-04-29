import numpy as np
import attr

from eradiate.solvers.onedim import *
from eradiate.solvers.onedim import _make_distant, _make_default_scene


def test_make_sensor(variant_scalar_mono):
    # Without units
    sensor = _make_distant(45., 0., 32)
    assert sensor.to_xml() == \
        '<sensor type="distant">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="32"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '<string name="pixel_format" value="luminance"/>' \
        '<rfilter type="box"/>' \
        '</film>' \
        '<vector name="direction" value="-0.7071067811865475, -0.0, -0.7071067811865476"/>' \
        '<point name="target" value="0, 0, 0"/>' \
        '</sensor>'

    # With degrees
    sensor = _make_distant(45. * ureg.deg, 0., 32)
    assert sensor.to_xml() == \
        '<sensor type="distant">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="32"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '<string name="pixel_format" value="luminance"/>' \
        '<rfilter type="box"/>' \
        '</film>' \
        '<vector name="direction" value="-0.7071067811865475, -0.0, -0.7071067811865476"/>' \
        '<point name="target" value="0, 0, 0"/>' \
        '</sensor>'

    # With radian
    sensor = _make_distant(0.25 * np.pi * ureg.rad, 0., 32)
    assert sensor.to_xml() == \
        '<sensor type="distant">' \
        '<sampler type="independent">' \
        '<integer name="sample_count" value="32"/>' \
        '</sampler>' \
        '<film type="hdrfilm">' \
        '<integer name="width" value="1"/>' \
        '<integer name="height" value="1"/>' \
        '<string name="pixel_format" value="luminance"/>' \
        '<rfilter type="box"/>' \
        '</film>' \
        '<vector name="direction" value="-0.7071067811865475, -0.0, -0.7071067811865476"/>' \
        '<point name="target" value="0, 0, 0"/>' \
        '</sensor>'


def test_onedimsolver(variant_scalar_mono):
    # Construct
    solver = OneDimSolver()
    assert solver.scene.to_xml() == _make_default_scene().to_xml()

    # Run simulation with default parameters (and check if result array is cast to scalar)
    assert solver.run() == 0.1591796875

    # Run simulation with array of vzas (and check if result array is squeezed)
    result = solver.run(vza=np.linspace(0, 90, 91), spp=32)
    assert result.shape == (91,)
    assert np.all(result == 0.1591796875)

    # Run simulation with array of vzas and vaas
    result = solver.run(vza=np.linspace(0, 90, 11),
                        vaa=np.linspace(0, 180, 11),
                        spp=32)
    assert result.shape == (11, 11)
    assert np.all(result == 0.1591796875)


def test_rayleigh_homogeneous_solver(variant_scalar_mono):
    # Construct
    solver = RayleighHomogeneousSolver()
    assert solver.scene.to_xml() == \
        '<scene version="0.1.0">' \
        '<bsdf type="diffuse" id="surface_brdf">' \
        '<spectrum name="reflectance" value="0.5"/>' \
        '</bsdf>' \
        '<phase type="rayleigh" id="phase_rayleigh"/>' \
        '<medium type="homogeneous" id="medium_rayleigh">' \
        '<ref id="phase_rayleigh"/>' \
        '<spectrum name="sigma_t" value="1.2134616206590448e-05"/>' \
        '<spectrum name="albedo" value="1.0"/>' \
        '</medium>' \
        '<shape type="rectangle">' \
        '<transform name="to_world">' \
        '<scale value="1.0"/>' \
        '</transform>' \
        '<ref id="surface_brdf"/>' \
        '</shape>' \
        '<shape type="cube">' \
        '<transform name="to_world">' \
        '<scale value="1.0, 1.0, 1.0"/>' \
        '<translate value="0.0, 0.0, 1.0"/>' \
        '</transform>' \
        '<ref name="interior" id="medium_rayleigh"/>' \
        '<bsdf type="null"/>' \
        '</shape>' \
        '<emitter type="directional">' \
        '<vector name="direction" value="0, 0, -1"/>' \
        '<spectrum name="irradiance" value="1.0"/>' \
        '</emitter>' \
        '<integrator type="volpath"/>' \
        '</scene>'
    solver.scene.instantiate()
