"""
Visualise spectral response function data sets
==============================================
"""

# %%
# This tutorial illustrates how to visualise the data from a
# :ref:`spectral response function data set <sec-user_guide-data-srf>`.

import matplotlib.pyplot as plt
import eradiate

# %%
# We will plot the spectral response function from the band 5 of the OLCI
# instrument onboard the Sentinel-3A platform.
# Open the data set using the
# :meth:`eradiate.data.open` method.
# Spectral response function data sets belong to the
# ``spectral_response_function`` category.
# Spectral response function data set identifiers are defined according to the
# :ref:`following convention <sec-user_guide-data-srf-naming_convention>`.
# In this example, the data set that we want to open has the
# ``sentinel_3a-olci-5`` identifier:

ds = eradiate.data.open(category="spectral_response_function",
                        id="sentinel_3a-olci-5")

# %%
# Plot the spectral response function data by calling the ``plot`` method
# of the ``srf`` data variable:

ds.srf.plot()
plt.show()

# %%
# Plot the spectral response function uncertainties by calling the ``plot``
# method of the ``srf_u`` data variable:

ds.srf_u.plot()
plt.show()

# %%
# .. note::
#    Not all calibration teams provide the uncertainties on the spectral
#    response function data.
#    In those cases, the above code will produce an empty plot.
