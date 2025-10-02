# Derived from https://github.com/nengo/pytest-plt/
# The original project is MIT-licensed

import os
import pickle
import re

import pytest
from matplotlib import pyplot as mpl_plt


class Mock:
    multi_functions = {}

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    def __getitem__(self, key):
        return Mock()

    def __iter__(self):
        return iter([])

    def __mul__(self, other):
        return 1.0

    @classmethod
    def __getattr__(cls, name):
        if name in ("__file__", "__path__"):
            return "/dev/null"
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        elif name in cls.multi_functions:
            return lambda *args, **kwargs: tuple(
                Mock() for _ in range(cls.multi_functions[name])
            )
        else:
            return Mock()


class PltMock(Mock):
    multi_functions = {"subplots": 2}


class Recorder:
    def __init__(self, dirname, nodeid, filename_drop=None):
        self.dirname = dirname
        self.nodeid = nodeid
        self.saved = None
        self.filename_drop = [] if filename_drop is None else filename_drop

    @property
    def record(self):
        return self.dirname is not None

    @property
    def dirname(self):
        return self._dirname

    @dirname.setter
    def dirname(self, _dirname):
        if _dirname is not None:
            _dirname = os.path.normpath(_dirname)
            if not os.path.exists(_dirname):
                os.makedirs(_dirname)
        self._dirname = _dirname

    def get_filename(self, ext=""):
        # Flatten filestructure (replace folders with dots in filename).
        # The nodeid should only contain /, but to be safe we also replace \\.
        # We also replace : with - because Windows does not allow colons in filenames.
        filename = self.nodeid.replace("/", ".").replace("\\", ".").replace(":", "-")

        # Drop parts of filename matching the given regexs
        for pattern in self.filename_drop:
            match = re.search(pattern, filename)
            while match is not None:
                span = match.span()
                filename = filename[: span[0]] + filename[span[1] :]
                match = re.search(pattern, filename)

        return f"{filename}.{ext}"

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def save(self, path):
        self.saved = path


class Plotter(Recorder):
    def __enter__(self):
        if self.record:
            self.plt = mpl_plt
        else:
            self.plt = PltMock()
        self.plt.saveas = self.get_filename(ext="pdf")
        return self.plt

    def __exit__(self, type, value, traceback):
        if self.record:
            if self.plt.saveas is None:
                del self.plt.saveas
                self.plt.close("all")
                return

            self.save(os.path.join(self.dirname, self.plt.saveas))
            self.plt.close("all")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith(".pkl") or path.endswith(".pickle"):
            with open(path, "wb") as fh:
                pickle.dump(self.plt.gcf(), fh)
        else:
            savefig_kw = {"bbox_inches": "tight"}
            if hasattr(self.plt, "bbox_extra_artists"):
                savefig_kw["bbox_extra_artists"] = self.plt.bbox_extra_artists
            self.plt.savefig(path, **savefig_kw)

        super().save(path)


@pytest.fixture
def plt(request):
    """
    A pyplot-compatible plotting interface.

    Use this to create plots in your tests using the ``matplotlib.pyplot``
    interface.

    This will keep saved plots organized in a simulator-specific folder,
    with an automatically generated name. ``savefig()`` and ``close()`` will
    automatically be called when the test function completes.

    If you need to override the default filename, set ``plt.saveas`` to
    the desired filename. Be sure to include a file extension, as it will
    be used as is.
    """
    # Read plt_filename_drop from .ini config file
    filename_drop = request.config.getini("plt_filename_drop")
    filename_drop = [s for s in filename_drop.split("\n") if len(s) > 0]

    # Read plt_dirname from .ini config file
    default_dirname = request.config.getini("plt_dirname")

    # Read dirname from command line, which takes precedence over .ini config
    dirname = request.config.getvalue("plots")
    if not isinstance(dirname, str) and dirname:
        dirname = default_dirname
    elif not dirname:
        dirname = None  # --plots argument not provided, so disable plots

    plotter = Plotter(dirname, request.node.nodeid, filename_drop=filename_drop)

    def _finalize():
        plotter.__exit__(None, None, None)
        if plotter.saved is not None:
            request.node.user_properties.append(("plt_saved", plotter.saved))

    request.addfinalizer(_finalize)
    return plotter.__enter__()  # pylint: disable=unnecessary-dunder-call
