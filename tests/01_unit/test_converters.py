import eradiate.converters as converters


def test_passthrough():
    def converter(v):
        return int(v)

    c = converters.passthrough(lambda x: isinstance(x, str))(converter)
    assert c(1.0) == 1
    assert c("1") == "1"

    c = converters.passthrough_type(str)(converter)
    assert c(1.0) == 1
    assert c("1") == "1"


def test_resolve_keyword():
    c = converters.resolve_keyword(lambda x: f"tests/thermoprops/{x}.nc")
    path = c("cams_lybia4_2005-04-01")
    assert path.is_absolute()
    assert path.is_file()
