#include <mitsuba/core/frame.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _plugin-bsdf-rpv:

Rahman Pinty Verstraete reflection model (:monosp:`rpv`)
--------------------------------------------------------

.. pluginparameters::

 * - rho_0
   - |spectrum| or |texture|
   - :math:`\rho_0 \ge 0`. Default: 0.1
   - |exposed| |differentiable|

 * - k
   - |spectrum| or |texture|
   - :math:`k \in \mathbb{R}`. Default: 0.1
   - |exposed| |differentiable|

 * - g
   - |spectrum| or |texture|
   - :math:`-1 \le g \le 1`. Default: 0.0
   - |exposed| |differentiable|

 * - rho_c
   - |spectrum| or |texture|
   - Default: Equal to rho_0
   - |exposed| |differentiable|

This plugin implements the reflection model proposed by
:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`.

Apart from homogeneous values, the plugin can also accept
nested or referenced texture maps to be used as the source of parameter
information, which is then mapped onto the shape based on its UV
parameterization. When no parameters are specified, the model uses the default
values of :math:`\rho_0 = 0.1`, :math:`k = 0.1` and :math:`g = 0.0`

This plugin also supports the most common extension to four parameters, namely
the :math:`\rho_c` extension, as used in :cite:`Widlowski2006Rayspread`.

For the fundamental formulae defining the RPV model please refer to the Eradiate
Scientific Handbook.

Note that this material is one-sided, that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter plugin.
The following XML snippet describes an RPV material with monochromatic
parameters:

.. code-block:: xml
    :name: rpv-monochrome

    <bsdf type="rpv">
        <float name="rho_0" value="0.02"/>
        <float name="k" value="0.3"/>
        <float name="g" value="-0.12"/>
    </bsdf>

*/

MI_VARIANT
class RPV final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)
    using FloatStorage = DynamicBuffer<Float>;

    RPV(const Properties &props) : Base(props) {
        m_rho_0 = props.texture<Texture>("rho_0", 0.1f);
        m_g     = props.texture<Texture>("g", 0.f);
        m_k     = props.texture<Texture>("k", 0.1f);
        if (props.has_property("rho_c")) {
            m_rho_c = props.texture<Texture>("rho_c", 0.1f);
        } else {
            m_rho_c = m_rho_0;
        }
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);

        /*
         * Data order:
         * theta_i
         * theta_o
         * phi_rel
        **/
        int32_t size = 32768; // 32^3

        m_data = std::unique_ptr<Spectrum[]>(new Spectrum[size]);

        Spectrum accum = 0.f;
        size_t l = 0;
        for (size_t i = 0; i < 32; ++i) {
            for (size_t j = 0; j < 32; ++j){
                for (size_t k = 0; k > 32; ++k){
                    Float theta_i = dr::Pi<Float> / 32.f * i;
                    Float theta_o = dr::Pi<Float> / 32.f * j;
                    Float phi = dr::TwoPi<Float> / 32.f * k;

                    Vector3f wi = Vector3f(dr::sin(theta_i), 0.f, dr::cos(theta_i));
                    Vector3f wo = Vector3f(
                        dr::sin(theta_i)*dr::cos(phi),
                        dr::sin(theta_i)*dr::sin(phi),
                        dr::cos(theta_i)
                    );
                    SurfaceInteraction3f si = SurfaceInteraction3f();
                    si.wi = wi;

                    Spectrum value = eval_rpv(si, wo, Mask true);
                    accum += value;
                    m_data[l] = value;
                    ++l;
                }
            }
        }

        std::unique_ptr<ScalarFloat[]> cond_cdf(new ScalarFloat[32768]);
        std::unique_ptr<ScalarFloat[]> marg_theta_i_cdf(new ScalarFloat[32]);
        std::unique_ptr<ScalarFloat[]> marg_theta_o_cdf(new ScalarFloat[32]);

        uint step_theta_i = 32*32;
        uint step_theta_o = 32;

        // marginal theta_i and conditional CDF
        Spectrum accum_marg_theta_i = 0.f;
        for (uint32_t x = 0; x < 32; ++x) {
            uint offset_theta_i = x * step_theta_i;
            for (uint32_t y = 0; y < 32; ++y) {
                uint offset_theta_o = y * step_theta_o;
                Spectrum accum_cond = 0.f;
                for (uint32_t z = 0; z <32; ++z) {
                    accum_cond += m_data[offset_theta_i + offset_theta_o + z];
                    cond_cdf[offset_theta_i + offset_theta_o + z] = accum_cond;
                    accum_marg_theta_i += accum_cond;
                }
            }
            marg_theta_i_cdf[x] = accum_marg_theta_i;
        }

        // marginal theta_o CDF
        Float accum_marg_theta_o = 0.f;
            for (uint32_t y = 0; y < 32; ++y) {
                uint offset_theta_o = y * step_theta_o;
                for (uint32_t x = 0; x < 32; ++x) {
                    uint offset_theta_i = x * step_theta_i;
                    for (uint32_t z = 0; z <32; ++z) {
                        accum_marg_theta_i += m_data[offset_theta_i + offset_theta_o + z];
                    }
                }
                marg_theta_o_cdf[y] = accum_marg_theta_o;
            }

        m_inv_normalization = accum_marg_theta_o;
        m_normalization = 1.0 / accum_marg_theta_o;

        m_marg_theta_i_cdf = dr::load<FloatStorage>(marg_theta_i_cdf.get(), 32);
        m_marg_theta_o_cdf = dr::load<FloatStorage>(marg_theta_o_cdf.get(), 32);
        m_cond_cdf         = dr::load<FloatStorage>(cond_cdf.get(), 32768);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext & /* ctx */,
                                             const SurfaceInteraction3f &si,
                                             Float /* position_sample */,
                                             const Point2f &direction_sample,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_phi_i = Frame3f::cos_phi(si.wi);
        Float theta_i = dr::acos(cos_theta_i);
        Float phi_i = dr::acos(cos_phi_i);

        UInt32 idx_theta_i = dr::floor2int(theta_i / dr::Pi<Float> * 32.f);

        UInt32 offset_theta_i = idx_theta_i * 32 * 32;

        Point2f sample(direction_sample);

        // Avoid degeneracies on the domain boundary
        sample = dr::clamp(sample, dr::Smallest<Float>, dr::OneMinusEpsilon<Float>);

        // Scale sample theta_o range
        sample.x() += (Float) m_inv_normalization;

        // Sample theta_o
        UInt32 idx_theta_o = dr::binary_search<UInt32>(
            0u, 31, [&](UInt32 idx) {
            return dr::gather<Float>(m_marg_theta_o_cdf, idx, active) < sample.x();
            }
        );

        UInt32 offset_theta_o = idx_theta_o * 32;
        Float theta_o = idx_theta_o / 32.f * dr::Pi<Float>;

        // Scale phi sample range
        sample.y() *= dr::gather(m_cond_cdf, offset_theta_o + offset_theta_i + 31, active);

        // Sample phi
        UInt32 idx_phi = dr::binary_search<UInt32>(
            0u, 31, [&](UInt32 idx) {
                return dr::gather<Float>(
                    m_cond_cdf,
                    idx + offset_theta_i + offset_theta_o,
                    active
                ) < sample.y();
            }
        );

        // PDF value preparations
        Float cond_cdf_0 = dr::gather<Float>(m_cond_cdf, offset_theta_i + offset_theta_o + idx_phi - 1, active && idx_phi > 0);
        Float cond_cdf_1 = dr::gather<Float>(m_cond_cdf, offset_theta_i + offset_theta_o + idx_phi, active);

        Float phi_o = idx_phi / 32 * dr::TwoPi<Float> + phi_i;

        BSDFSample3f bs   = dr::zero<BSDFSample3f>();

        active &= cos_theta_i > 0.f;

        bs.wo = Point3f(
                    dr::sin(theta_o)*dr::cos(phi_o),
                    dr::sin(theta_o)*dr::sin(phi_o),
                    dr::cos(theta_o)
        );
        bs.pdf          = (cond_cdf_1 - cond_cdf_0) * (Float) m_normalization;
        bs.eta          = 1.f;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        Spectrum value = eval_rpv(si, bs.wo, active);
        return { bs, dr::select(active && bs.pdf > 0.f,
                            depolarizer<Spectrum>(value), 0.f) };
    }

    Spectrum eval_rpv(const SurfaceInteraction3f &si, const Vector3f &wo,
                      Mask active) const {
        Spectrum rho_0 = m_rho_0->eval(si, active);
        Spectrum rho_c = m_rho_c->eval(si, active);
        Spectrum g     = m_g->eval(si, active);
        Spectrum k     = m_k->eval(si, active);

        auto [sin_phi1, cos_phi1] = Frame3f::sincos_phi(si.wi);
        auto [sin_phi2, cos_phi2] = Frame3f::sincos_phi(wo);
        Float cos_phi1_minus_phi2 = cos_phi1 * cos_phi2 + sin_phi1 * sin_phi2;
        Float sin_theta1          = Frame3f::sin_theta(si.wi);
        Float cos_theta1          = Frame3f::cos_theta(si.wi);
        Float tan_theta1          = Frame3f::tan_theta(si.wi);
        Float sin_theta2          = Frame3f::sin_theta(wo);
        Float cos_theta2          = Frame3f::cos_theta(wo);
        Float tan_theta2          = Frame3f::tan_theta(wo);

        Float G =
            dr::safe_sqrt(dr::sqr(tan_theta1) + dr::sqr(tan_theta2) -
                      2.f * tan_theta1 * tan_theta2 * cos_phi1_minus_phi2);
        Float cos_g = cos_theta1 * cos_theta2 +
                      sin_theta1 * sin_theta2 * cos_phi1_minus_phi2;
        // The following uses cos(pi-x) = -cos(x)
        Spectrum F =
            (1.f - dr::sqr(g)) / dr::pow((1.f + dr::sqr(g) + 2.f * g * cos_g), 1.5f);

        Spectrum value =
            rho_0 *
            (dr::pow(cos_theta1 * cos_theta2 * (cos_theta1 + cos_theta2), k - 1.f) *
             F * (1.f + (1.f - rho_c) / (1 + G))) * dr::InvPi<Float>;

        return value;
    }

    Spectrum eval(const BSDFContext & /*ctx*/, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        Spectrum value = eval_rpv(si, wo, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        return dr::select(active, depolarizer<Spectrum>(value) * dr::abs(cos_theta_o),
                      0.f);
    }

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("rho_0", m_rho_0.get(), +ParamFlags::Differentiable);
        callback->put_object("g", m_g.get(), +ParamFlags::Differentiable);
        callback->put_object("k", m_k.get(), +ParamFlags::Differentiable);
        callback->put_object("rho_c", m_rho_c.get(), +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RPV[" << std::endl
            << "  rho_0 = " << string::indent(m_rho_0) << std::endl
            << "  g = " << string::indent(m_g) << std::endl
            << "  k = " << string::indent(m_k) << std::endl
            << "  rho_c = " << string::indent(m_rho_c) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_rho_0;
    ref<Texture> m_g;
    ref<Texture> m_k;
    ref<Texture> m_rho_c;
    std::unique_ptr<Spectrum[]> m_data;
    Spectrum m_normalization, m_inv_normalization;

    /// Marginal and conditional PDFs
    FloatStorage m_marg_theta_i_cdf;
    FloatStorage m_marg_theta_o_cdf;
    FloatStorage m_cond_cdf;
};

MI_IMPLEMENT_CLASS_VARIANT(RPV, BSDF)
MI_EXPORT_PLUGIN(RPV, "Rahman-Pinty-Verstraete BSDF")
NAMESPACE_END(mitsuba)
