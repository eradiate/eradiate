#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _plugin-bsdf-bilambertian:

Bi-Lambertian material (:monosp:`bilambertian`)
-----------------------------------------------

.. pluginparameters::

 * - reflectance
   - |spectrum| or |texture|
   - Specifies the diffuse reflectance of the material (Default: 0.5)
 * - transmittance
   - |spectrum| or |texture|
   - Specifies the diffuse transmittance of the material (Default: 0.5)

The bi-Lambertian material scatters light diffusely into the entire sphere.
The reflectance specifies the amount of light scattered into the incoming
hemisphere, while the transmittance specifies the amount of light scattered
into the outgoing hemisphere. This material is two-sided.

.. note::

   This material is not designed for realistic rendering, but rather for
   large-scale simulation of atmospheric radiative transfer over vegetated
   surfaces.

*/

template <typename Float, typename Spectrum>
class BiLambertian final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    BiLambertian(const Properties &props) : Base(props) {
        m_reflectance   = props.texture<Texture>("reflectance", .5f);
        m_transmittance = props.texture<Texture>("transmittance", .5f);

        m_components.push_back(BSDFFlags::DiffuseReflection |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);
        m_components.push_back(BSDFFlags::DiffuseTransmission |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);

        m_flags = m_components[0] | m_components[1];
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely(ek::none_or<false>(active) ||
                     (!has_reflect && !has_transmit)))
            return { ek::zero<BSDFSample3f>(), UnpolarizedSpectrum(0.f) };

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Vector3f wo       = warp::square_to_cosine_hemisphere(sample2);

        BSDFSample3f bs = ek::zero<BSDFSample3f>();
        UnpolarizedSpectrum value(0.f);

        // Select the lobe to be sampled
        UnpolarizedSpectrum r              = m_reflectance->eval(si, active),
                            t              = m_transmittance->eval(si, active);
        Float reflection_sampling_weight   = hmean(r / (r + t)),
              transmission_sampling_weight = 1.f - reflection_sampling_weight;

        // Handle case where r = t = 0
        ek::masked(reflection_sampling_weight,
                   ek::isnan(reflection_sampling_weight))   = 0.f;
        ek::masked(transmission_sampling_weight,
                   ek::isnan(transmission_sampling_weight)) = 0.f;

        Mask selected_r = (sample1 < reflection_sampling_weight) && active,
             selected_t = (sample1 >= reflection_sampling_weight) && active;

        // Evaluate
        value = ek::select(active, Float(1.f), 0.f);
        value[selected_r] *= r / reflection_sampling_weight;
        value[selected_t] *= t / transmission_sampling_weight;

        // Compute PDF
        bs.pdf =
            ek::select(active, warp::square_to_cosine_hemisphere_pdf(wo), 0.f);
        bs.pdf =
            ek::select(selected_r, bs.pdf * reflection_sampling_weight, bs.pdf);
        bs.pdf = ek::select(selected_t, bs.pdf * transmission_sampling_weight,
                            bs.pdf);

        // Set other interaction fields
        bs.eta               = 1.f;
        bs.sampled_component = ek::select(selected_r, UInt32(0), UInt32(1));
        bs.sampled_type =
            ek::select(selected_r, UInt32(+BSDFFlags::DiffuseReflection),
                       UInt32(+BSDFFlags::DiffuseTransmission));

        // Flip the outgoing direction if the incoming comes from "behind"
        wo = ek::select(cos_theta_i > 0, wo, Vector3f(wo.x(), wo.y(), -wo.z()));

        // Flip the outgoing direction if transmission was selected
        bs.wo = ek::select(selected_r, wo, Vector3f(wo.x(), wo.y(), -wo.z()));

        return { bs, ek::select(active && bs.pdf > 0.f,
                                depolarizer<Spectrum>(value), 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely((!has_reflect && !has_transmit) ||
                     ek::none_or<false>(active)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        UnpolarizedSpectrum result(0.f);

        if (has_reflect) {
            // If reflection is activated, compute reflection for relevant
            // directions
            auto is_reflect =
                Mask(ek::eq(ek::sign(cos_theta_i), ek::sign(cos_theta_o))) && active;
            result[is_reflect] = m_reflectance->eval(si, is_reflect);
        }

        if (has_transmit) {
            // If transmission is activated, compute transmission for relevant
            // directions
            auto is_transmit =
                Mask(ek::neq(ek::sign(cos_theta_i), ek::sign(cos_theta_o))) && active;
            result[is_transmit] = m_transmittance->eval(si, is_transmit);
        }

        result[active] *= (ek::InvPi<Float> * abs(cos_theta_o));

        return ek::select(active, result, 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely(ek::none_or<false>(active) ||
                     (!has_reflect && !has_transmit)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Ensure that uncoming direction is in upper hemisphere
        Vector3f wo_flip{ wo.x(), wo.y(), abs(cos_theta_o) };

        Float result = ek::select(
            active, warp::square_to_cosine_hemisphere_pdf(wo_flip), 0.f);

        UnpolarizedSpectrum r              = m_reflectance->eval(si, active),
                            t              = m_transmittance->eval(si, active);
        Float reflection_sampling_weight   = hmean(r / (r + t)),
              transmission_sampling_weight = 1.f - reflection_sampling_weight;

        // Handle case where r = t = 0
        ek::masked(reflection_sampling_weight,
                   ek::isnan(reflection_sampling_weight))   = 0.f;
        ek::masked(transmission_sampling_weight,
                   ek::isnan(transmission_sampling_weight)) = 0.f;

        if (has_reflect) {
            auto is_reflect =
                Mask(ek::eq(ek::sign(cos_theta_i), ek::sign(cos_theta_o))) && active;
            ek::masked(result, is_reflect) *= reflection_sampling_weight;
        }

        if (has_transmit) {
            auto is_transmit =
                Mask(ek::neq(ek::sign(cos_theta_i), ek::sign(cos_theta_o))) && active;
            ek::masked(result, is_transmit) *= transmission_sampling_weight;
        }

        return result;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("reflectance", m_reflectance.get());
        callback->put_object("transmittance", m_transmittance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Bilambertian[" << std::endl
            << "  reflectance = " << string::indent(m_reflectance) << std::endl
            << "  transmittance = " << string::indent(m_transmittance)
            << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
    ref<Texture> m_transmittance;
};

MTS_IMPLEMENT_CLASS_VARIANT(BiLambertian, BSDF)
MTS_EXPORT_PLUGIN(BiLambertian, "Bi-Lambertian material")
NAMESPACE_END(mitsuba)
