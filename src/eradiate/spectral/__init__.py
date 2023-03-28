"""Data structures for working along the spectral dimension.

This includes in particular:

* :class:`.SpectralIndex`: a data structure used to evaluate spectral quantities
* :class:`.WavelengthSet`: a set of wavelength values at which an experiment is
  run
* :class:`.BinSet`: a set of wavelength bins at which an experiment is run
"""
import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
