import setuptools

setuptools.setup(
    version_config={
        "dev_template": "{sha}",
        "dirty_template": "{sha}",
    },
    setup_requires=['setuptools-git-versioning'],
)
