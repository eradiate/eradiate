"""Utilities for system tests"""

import enoki as ek
import numpy as np
from eradiate.kernel.core import Bitmap

def aov_to_variance(bmp):
    # AVOs from the moment integrator are in XYZ (float32)
    if isinstance(bmp, Bitmap):
        split = bmp.split()
        img = np.array(split[1][1], copy=False)
        img_m2 = np.array(split[2][1], copy=False)
    else:
        ...
    return img, img_m2 - img * img


def z_test(mean, sample_count, reference, reference_var, significance_level=0.01):
    # Sanitize the variance images
    reference_var = np.maximum(reference_var, 1e-4)

    # Compute Z statistic
    z_stat = np.abs(mean - reference) * np.sqrt(sample_count / reference_var)

    # Cumulative distribution function of the standard normal distribution
    def stdnormal_cdf(x):
        shape = x.shape
        cdf = (1.0 - ek.erf(-x.flatten() / ek.sqrt(2.0))) * 0.5
        return np.array(cdf).reshape(shape)

    # Compute p-value
    p_value = 2.0 * (1.0 - stdnormal_cdf(z_stat))

    # Apply the Sidak correction term, since we'll be conducting multiple independent
    # hypothesis tests. This accounts for the fact that the probability of a failure
    # increases quickly when several hypothesis tests are run in sequence.
    pixel_count = len(np.flatten(reference))
    alpha = 1.0 - (1.0 - significance_level) ** (1.0 / pixel_count)

    success = (p_value > alpha)

    return success