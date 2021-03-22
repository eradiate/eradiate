import setuptools

setuptools.setup(
    version_config={
        "dev_template": "{tag}",
        "dirty_template": "{tag}",
    },
    setup_requires=['setuptools-git-versioning'],
)
