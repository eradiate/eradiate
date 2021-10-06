.. _sec-reference-kernel:

Kernel [eradiate.kernel]
========================

Eradiate's computational kernel is based on the Mitsuba 2 rendering system.
Eradiate imports the :mod:`mitsuba` module and makes a basic check to detect if
it is not our customised version. The :mod:`eradiate.kernel` subpackage provides
convenience shortcuts to the Mitsuba API, but everything done here can also be
done with the regular Mitsuba Python bindings.

Customised Mitsuba 2 [eradiate.kernel.mitsuba]
----------------------------------------------

.. link-button:: https://eradiate-kernel.readthedocs.io
   :type: url
   :text: Go to Eradiate Kernel docs
   :classes: btn-outline-primary btn-block

Volume data file I/O routines [eradiate.kernel.gridvolume]
----------------------------------------------------------

.. currentmodule:: eradiate.kernel.gridvolume

.. autosummary::
   :toctree: generated/autosummary/

   write_binary_grid3d
   read_binary_grid3d
