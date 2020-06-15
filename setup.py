from setuptools import setup, find_packages

setup(
    name="eradiate",
    version="0.0.1",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[],
    entry_points={
        "console_scripts": [
            "ertrayleigh = eradiate.scripts.ertrayleigh:cli"
        ]
    }
)