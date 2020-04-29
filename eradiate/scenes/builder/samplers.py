import attr

from .base import Int, Plugin
from .util import parameter


@attr.s
class Sampler(Plugin):
    """
    Abstract base class for all sampler plugins.
    """
    _tag = "sampler"


_independent_params = {
    name: parameter(type, name)
    for name, type in [
        ("sample_count", Int),
        ("seed", Int),
    ]
}


@attr.s(these=_independent_params)
class Independent(Sampler):
    _type = "independent"
    _params = Sampler._params + list(_independent_params)
