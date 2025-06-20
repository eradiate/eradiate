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
