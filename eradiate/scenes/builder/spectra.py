import attr

from .base import Object
from .util import seq_to_str
from ...util import ensure_array


@attr.s
class Spectrum(Object):
    """
    Base class for spectra.
    """

    _tag = "spectrum"
    value = attr.ib(converter=ensure_array)

    def __add__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(value=self.value + other.value)
        elif isinstance(other, (float, int)):
            return Spectrum(value=self.value + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: "
                            f"'{type(self).__name__}' and "
                            f"'{type(other).__name__}'")

    def __radd__(self, other):
        return Spectrum(value=self.value + other)

    def __sub__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(value=self.value - other.value)
        elif isinstance(other, (float, int)):
            return Spectrum(value=self.value - other)
        else:
            raise TypeError(f"unsupported operand type(s) for -: "
                            f"'{type(self).__name__}' and "
                            f"'{type(other).__name__}'")

    def __rsub__(self, other):
        return Spectrum(value=other - self.value)

    def __neg__(self):
        return Spectrum(value=-self.value)

    def __mul__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(value=self.value * other.value)
        elif isinstance(other, (float, int)):
            return Spectrum(value=self.value * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: "
                            f"'{type(self).__name__}' and "
                            f"'{type(other).__name__}'")

    def __rmul__(self, other):
        return Spectrum(value=self.value * other)

    def __truediv__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(value=self.value / other.value)
        elif isinstance(other, (float, int)):
            return Spectrum(value=self.value / other)
        else:
            raise TypeError(f"unsupported operand type(s) for /: "
                            f"'{type(self).__name__}' and "
                            f"'{type(other).__name__}'")

    def __rtruediv__(self, other):
        return Spectrum(value=other / self.value)

    def __inv__(self):
        return Spectrum(value=1. / self.value)

    def to_etree(self):
        e = super().to_etree()
        e.set("value", seq_to_str(self.value))
        return e
