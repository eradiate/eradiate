# IPython extension. Must be imported by top-level module.


def load_ipython_extension(ipython):
    from eradiate.notebook import install

    install()
