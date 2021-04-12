import numpy as np
import enoki as ek

from eradiate.kernel.core import Bitmap, Struct

def z_test(mean, sample_count, reference, reference_var):
    # Sanitize the variance images
    # reference_var = np.maximum(reference_var, 1e-4)

    # Compute Z statistic
    z_stat = np.abs(mean - reference) * np.sqrt(sample_count / reference_var)

    # Cumulative distribution function of the standard normal distribution
    def stdnormal_cdf(x):
        shape = x.shape
        cdf = (1.0 - ek.erf(-x.flatten() / ek.sqrt(2.0))) * 0.5
        return np.array(cdf).reshape(shape)

    # Compute p-value
    p_value = 2.0 * (1.0 - stdnormal_cdf(z_stat))

    return p_value

def bitmap_extract(bmp):
    split = bmp.split()
    img = np.squeeze(np.array(split[2][1].convert(Bitmap.PixelFormat.Y, Struct.Type.Float32, srgb_gamma=False), copy=False))
    img_m2 = np.squeeze(np.array(split[1][1].convert(Bitmap.PixelFormat.Y, Struct.Type.Float32, srgb_gamma=False), copy=False))
    return img, img_m2 - (img * img)