""" Unit system-related components. """

import pint

#: Unit registry common to all Eradiate components.
ureg = pint.UnitRegistry()

#: Alias to :data:`ureg`'s ``Quantity`` member.
Q_ = ureg.Quantity
