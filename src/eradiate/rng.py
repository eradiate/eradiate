"""
Components related with pseudo-random number generation.

Inspired by `SeedBank <https://github.com/lenskit/seedbank>`__.
"""

from __future__ import annotations

import attrs
import numpy as np
import numpy.random


@attrs.define
class SeedState:
    """
    Manage a root seed and facilities to derive seeds.
    """

    _seed: np.random.SeedSequence | None = attrs.field(
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
            Value used to initialize the internal seed sequence. If unset, the
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
        Return a default Numpy RNG initialized with a generated seed.

        Returns
        -------
        numpy.random.Generator
            Initialized RNG.
        """
        seed = self.next(1)[0]
        return np.random.default_rng(seed=seed)


#: Deterministic root seed state (see :class:`.SeedState`).
root_seed_state = SeedState(0)
