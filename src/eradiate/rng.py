"""
Components related with pseudo-random number generation.
"""

import typing as t

import attr
import numpy as np
import numpy.random


@attr.s
class SeedState:
    """
    Manage a root seed and facilities to derive seeds.
    """

    _seed: t.Optional[np.random.SeedSequence] = attr.ib(
        default=None,
        converter=lambda x: x
        if isinstance(x, np.random.SeedSequence)
        else np.random.SeedSequence(x),
    )

    def reset(self, seed=None):
        """
        Reset the seed state.

        Parameters
        ----------
        seed : int or numpy.random.SeedSequence, optional
            Value used to initialise the internal seed sequence. If unset, the
            current seed sequence is reused, with its children spawned member
            reset.
        """
        if seed is not None:
            self._seed = (
                seed
                if isinstance(seed, np.random.SeedSequence)
                else np.random.SeedSequence(seed)
            )
        else:
            self._seed = np.random.SeedSequence(entropy=self._seed.entropy)

    def next(self, n: int = 1) -> np.ndarray:
        """
        Get the next *n* seed values.

        Parameters
        ----------
        n : int
            Number of seed values to generate.

        Returns
        -------
        ndarray
            Generated RNG seeds.
        """
        result = self._seed.spawn(1)[0].generate_state(n)
        return result

    def numpy_default_rng(self) -> numpy.random.Generator:
        """
        Return a default Numpy RNG initialised with a generated seed.

        Returns
        -------
        numpy.random.Generator
            Initialised RNG.
        """
        seed = self.next(1)[0]
        return np.random.default_rng(seed=seed)


#: Root seed state.
root_state = SeedState(0)
