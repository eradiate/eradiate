""" Unit system-related components. """

import pint

ureg = pint.UnitRegistry()  #: Unit registry common to all Eradiate components.
Q_ = ureg.Quantity  #: Alias to :data:`ureg`'s ``Quantity`` member.
