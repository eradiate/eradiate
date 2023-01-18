#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>

/**!

.. _phase-rayleigh_polarized:

Rayleigh polarized phase function (:monosp:`rayleigh_polarized`)
----------------------------------------------------------------

Scattering by particles that are much smaller than the wavelength
of light (e.g. individual molecules in the atmosphere) is well-approximated
by the Rayleigh scattering model. This plugin implements the polarized
version of this scattering model. This plugin is useful for simulating
scattering in planetary atmospheres.

This model has no parameters.

.. tabs::
    .. code-tab:: xml

        <phase type="rayleigh_polarized" />

    .. code-tab:: python
        'type': 'rayleigh_polarized'

*/

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RayleighPolarizedPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    RayleighPolarizedPhaseFunction(const Properties &props) : Base(props) {
        m_flags = +PhaseFunctionFlags::Anisotropic;
    }

    MI_INLINE Float eval_rayleigh(Float cos_theta) const {
        Float value;

        /* Non-zero elements */
        value(0, 0) = (3.f / 4.f) * (1.f + dr::sqr(cos_theta));
        value(1, 1) = value(0, 0);
        value(0, 1) = (-3.f / 4.f) * (1 - dr::sqr(cos_theta));
        value(1, 0) = value(0, 1);
        value(2, 2) = (3.f / 2.f) * cos_theta;
        value(3, 3) = value(2, 2);

        /* All other elements are zero */
        value(0, 2) = 0.f;
        value(0, 3) = 0.f;
        value(1, 2) = 0.f;
        value(1, 3) = 0.f;
        value(2, 0) = 0.f;
        value(2, 1) = 0.f;
        value(2, 3) = 0.f;
        value(3, 0) = 0.f;
        value(3, 1) = 0.f;
        value(3, 2) = 0.f;

        return (1.f / 4.f) * dr::InvPi<Float> * value
    }

    std::pair<Vector3f, Float> sample(const PhaseFunctionContext & /* ctx */,
                                      const MediumInteraction3f &mi,
                                      Float /* sample1 */,
                                      const Point2f &sample,
                                      Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float z   = 2.f * (2.f * sample.x() - 1.f);
        Float tmp = dr::sqrt(dr::sqr(z) + 1.f);
        Float A   = dr::cbrt(z + tmp);
        Float B   = dr::cbrt(z - tmp);
        Float cos_theta = A + B;
        Float sin_theta = dr::safe_sqrt(1.0f - dr::sqr(cos_theta));
        auto [sin_phi, cos_phi] = dr::sincos(dr::TwoPi<Float> * sample.y());

        auto wo = Vector3f{ sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };

        wo = mi.to_world(wo);
        Float pdf = eval_rayleigh(-cos_theta);
        return { wo, pdf };
    }

    Float eval(const PhaseFunctionContext & ctx,
               const MediumInteraction3f &mi, const Vector3f &wo,
               Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        /* Due to the coordinate system rotations for polarization-aware
           PhaseFunctions below we need to know the propagation direction of
           light.
           In the following, light arrives along `-wo_hat` and leaves along
           `+wi_hat`. */
        Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mi.wi,
                 wi_hat = ctx.mode == TransportMode::Radiance ? mi.wi : wo;

        return eval_rayleigh(dot(wo_hat, wi_hat));
    }

    std::string to_string() const override { return "RayleighPolarizedPhaseFunction[]"; }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(RayleighPolarizedPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(RayleighPolarizedPhaseFunction, "Rayleigh polarized phase function")
NAMESPACE_END(mitsuba)
