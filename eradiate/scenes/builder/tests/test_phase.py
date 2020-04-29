from eradiate.scenes.builder.base import Float
from eradiate.scenes.builder.phase import *
from eradiate.scenes.builder.util import load


def test_isotropic(variant_scalar_mono):
    """
    We test the creation of the XML snippet for an isotropic phase function,
    by creating an instance of the `Isotropic` class and calling its
    to_xml() method. We assert correctness by comparing to a reference
    string of XML.
    """

    i = Isotropic()
    assert i.to_xml() == \
        '<phase type="isotropic"/>'
    load(i)


def test_henyey_greenstein(variant_scalar_mono):
    """
    We test the creation of the XML snippet for a Henyey-Greenstein phase
    function, by creating an instance of the `Isotropic` class and calling
    its to_xml() method. We assert correctness by comparing to a reference
    string of XML.
    """

    hg = HenyeyGreenstein(g=Float(0.6))
    assert hg.to_xml() == \
        '<phase type="hg">' \
        '<float name="g" value="0.6"/>' \
        '</phase>'
    hg.instantiate()


def test_rayleigh():
    """
    We test the creation of the XML snippet for a Rayleigh phase function,
    by creating an instance of the `Isotropic` class and calling its to_xml()
    method. We assert correctness by comparing to a reference string of XML.
    """

    r = Rayleigh(delta=Float(0.2))
    assert r.to_xml() == \
        '<phase type="rayleigh">' \
        '<float name="delta" value="0.2"/>' \
        '</phase>'
    r.instantiate()
