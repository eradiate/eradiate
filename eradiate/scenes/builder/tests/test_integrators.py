from eradiate.scenes.builder.integrators import Direct, Path, VolPath


def test_direct(variant_scalar_mono):
    # Default init
    i = Direct()
    assert i.to_xml() == """<integrator type="direct"/>"""
    i.instantiate()

    # Init with parameters
    i = Direct(shading_samples=5)
    assert i.to_xml() == \
        '<integrator type="direct">' \
        '<integer name="shading_samples" value="5"/>' \
        '</integrator>'
    i.instantiate()


def test_path(variant_scalar_mono):
    # Default init
    i = Path()
    assert i.to_xml() == """<integrator type="path"/>"""
    i.instantiate()

    # Init with parameters
    i = Path(max_depth=5)
    assert i.to_xml() == \
        '<integrator type="path">' \
        '<integer name="max_depth" value="5"/>' \
        '</integrator>'
    i.instantiate()


def test_vol_path(variant_scalar_mono):
    # Default init
    i = VolPath()
    assert i.to_xml() == """<integrator type="volpath"/>"""
    i.instantiate()

    # Init with parameters
    i = VolPath(max_depth=-1)
    assert i.to_xml() == \
        '<integrator type="volpath">' \
        '<integer name="max_depth" value="-1"/>' \
        '</integrator>'
    i.instantiate()
