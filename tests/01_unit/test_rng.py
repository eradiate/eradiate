import numpy as np

from eradiate.rng import SeedState


def test_seed_state_construct():
    # A SeedState instance can be constructed without argument ...
    SeedState()

    # ... or from an integer seed
    SeedState(0)

    # ... or from an already initialized SeedSequence instance
    SeedState(np.random.SeedSequence(0))


def test_seed_state_generate():
    # We can generate pseudo random seeds using
    s = SeedState(0)
    n1 = s.next().squeeze()
    n2 = s.next().squeeze()
    assert n1 != n2

    # Seed arrays are also available
    assert s.next(10).shape == (10,)

    # We can reset the seed state to restart the seed sequence
    s.reset()
    assert s.next().squeeze() == n1

    # Resetting the seed state with a different seed changes the seed sequence
    s.reset(1)
    assert s.next().squeeze() != n1


def test_seed_state_rng():
    # We can conveniently request seeded RNGs from a SeedState instance
    s = SeedState()
    assert isinstance(s.numpy_default_rng(), np.random.Generator)
