from ...util import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "_core": [
            "AbstractHeterogeneousAtmosphere",
            "Atmosphere",
            "AtmosphereGeometry",
            "PlaneParallelGeometry",
            "SphericalShellGeometry",
            "atmosphere_factory",
        ],
        "_heterogeneous": ["HeterogeneousAtmosphere"],
        "_homogeneous": ["HomogeneousAtmosphere"],
        "_molecular_atmosphere": ["MolecularAtmosphere"],
        "_particle_dist": [
            "ArrayParticleDistribution",
            "ExponentialParticleDistribution",
            "GaussianParticleDistribution",
            "InterpolatorParticleDistribution",
            "ParticleDistribution",
            "UniformParticleDistribution",
            "particle_distribution_factory",
        ],
        "_particle_layer": ["ParticleLayer"],
    },
)

del lazy_loader
