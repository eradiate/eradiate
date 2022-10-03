"""
Special extension for tutorials.
"""
import datetime

from IPython.display import Markdown, display

from .. import __version__


def load_ipython_extension(ipython):
    display(
        Markdown(
            f"*Last updated: {datetime.datetime.now():%Y-%m-%d %H:%M} (eradiate v{__version__})*"
        )
    )
