.. _sec-user_guide-data-ckd:

Correlated-k distribution (CKD)
===============================

Eradiate supports simulation using the correlated-*k* distribution method. To
that end, various data are used and shipped as part of the software package.

Bin set definitions
-------------------

Bin set definitions are shipped in the form of NetCDF datasets. Below is the
minimal dataset structure required for Eradiate to be able to interpret bin
definitions data.

Coordinates (* means also dimension)
    * ``*bin`` (str): bin identifier
Variables
    * ``wmin [bin]`` (float): lower bound wavelength in nm
    * ``wmax [bin]`` (float): upper bound wavelength in nm
Metadata
    * ``quadrature_type`` (str): quadrature type
    * ``quadrature_n`` (int): number of quadrature points

Quadrature points
-----------------

CKD quadrature points are shipped in the form of NetCDF datasets. Below is the
minimal dataset structure required for Eradiate to be able to interpret CKD
quadrature points data sets.

Coordinates (* means also dimension)
    * ``*z`` (float): altitude
    * Multi-level index ``*bd`` (int). Corresponding coordinates are stored as
      NetCDF variables and data is reindexed upon loading:

      * ``bin [bd]`` (str): bin name
      * ``index [bd]`` (int): quadrature point index

Variables
    * ``k [z, bd]`` (float): quadrature point absorption coefficient value
      [:math:`\text{length}^{-1}`]

Metadata
    * ``bin_set`` (str): ID of the associated bin set

.. note::
   The impact in terms of storage saved or lost is yet to be evaluated but we
   consider it a minor concern. If this approach proves to be impractical, we
   may consider modifying the data format.
