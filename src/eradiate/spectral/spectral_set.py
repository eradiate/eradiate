from abc import ABC, abstractmethod


class SpectralSet(ABC):
    """
    An interface common to all spectral sets.
    """

    @property
    @abstractmethod
    def wavelengths(self):
        pass
