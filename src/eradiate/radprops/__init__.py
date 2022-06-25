"""
Atmospheric radiative properties calculation package.
"""


from ..util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": ["RadProfile", "rad_profile_factory"],
        "_us76_approx": ["US76ApproxRadProfile"],
        "_afgl1986": ["AFGL1986RadProfile"],
        "_array": ["ArrayRadProfile"],
    },
)

del lazy_loader
