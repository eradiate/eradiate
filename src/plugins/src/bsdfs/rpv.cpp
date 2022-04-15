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

        m_size = ScalarPoint3u(64, 64, 64);
        /*
         * Data order:
         * theta_i
         * theta_o
         * phi_rel
        **/
        m_data = std::unique_ptr<Float[]>(new Float[m_size.x()*m_size.y()*m_size.z()]);

        size_t l = 0;
        for (size_t i = 0; i < m_size.x(); ++i) {
            Float theta_i = dr::Pi<Float> / (2.f * Float(m_size.x())) * Float(i);
            for (size_t j = 0; j < m_size.y(); ++j){
                Float theta_o = dr::Pi<Float> / (2.f * Float(m_size.y())) * Float(j);
                for (size_t k = 0; k < m_size.z(); ++k){
                    Float phi = dr::TwoPi<Float> / Float(m_size.z()) * Float(k);

                    Vector3f wi = Vector3f(dr::sin(theta_i), 0.f, dr::cos(theta_i));
                    Vector3f wo = Vector3f(
                        dr::sin(theta_o)*dr::cos(phi),
                        dr::sin(theta_o)*dr::sin(phi),
                        dr::cos(theta_o)
                    );
                    SurfaceInteraction3f si = SurfaceInteraction3f();
                    si.wi = wi;
                    si.t = 0.f;
                    si.p = Point3f(0.f);
                    si.n = Point3f(0.f, 0.f, 1.f);

                    Spectrum value = eval_rpv(si, wo, true);
                    m_value_debug = theta_i;
                    m_data[l] = (Float) value.x();
                    ++l;
                }
            }
        }

        std::unique_ptr<Float[]> marg_phi_cdf(new Float[m_size.x() * m_size.z()]);
        std::unique_ptr<Float[]> cond_cdf(new Float[m_size.x()*m_size.y()*m_size.z()]);

        uint step_theta_i = m_size.y() * m_size.z();
        uint step_theta_o = m_size.z();
        uint step_phi = m_size.y();

        // marginal  and conditional CDFs
        for (uint32_t x = 0; x < m_size.x(); ++x) {
            uint offset_theta_i = x * step_theta_i;
            for (uint32_t z = 0; z < m_size.z(); ++z) {
                uint offset_phi = z * step_phi;
                Float accum_cond = 0.f;
                for (uint32_t y = 0; y < m_size.y(); ++y) {
                    uint offset_theta_o = y * step_theta_o;
                    accum_cond += m_data[offset_theta_i + offset_theta_o + z];
                    cond_cdf[offset_theta_i + offset_phi + y] = accum_cond;
                }
            }
        }
        m_inv_normalization = accum_marg_theta_i;
        m_normalization = 1.0 / accum_marg_theta_i;

        for (uint32_t x = 0; x < m_size.x(); ++x) {
            Float accum_marg_phi = 0.f;
            uint offset_x = m_size.z() * x;
            uint offset_theta_i = x * step_theta_i;
            for (uint32_t z = 0; z < m_size.z(); ++z) {
                for (uint32_t y = 0; y < m_size.y(); ++y) {
                    uint offset_theta_o = y * step_theta_o;
                    accum_marg_phi += m_data[offset_theta_i + offset_theta_o + z];
                }
                marg_phi_cdf[offset_x + z] = accum_marg_phi;
            }
        }

        m_marg_phi_cdf = dr::load<FloatStorage>(marg_phi_cdf.get(), m_size.x() * m_size.z());
        m_cond_cdf         = dr::load<FloatStorage>(cond_cdf.get(), dr::hprod(m_size));
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext & /* ctx */,
                                             const SurfaceInteraction3f &si_,
                                             Float /* position_sample */,
                                             const Point2f &direction_sample,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        //Normalize si.wi
        SurfaceInteraction3f si = si_;
        si.wi = dr::normalize(si.wi);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_phi_i = Frame3f::cos_phi(si.wi);
        Float theta_i = dr::acos(cos_theta_i);
        Float phi_i = dr::acos(cos_phi_i);

        UInt32 idx_theta_i = dr::floor2int<UInt32>(theta_i / (dr::Pi<Float> / 2.f) * Float(m_size.x()));

        // Avoid out of range indices
        idx_theta_i = dr::clamp(idx_theta_i, 0u, m_size.x() -1);

        UInt32 offset_theta_i_cond = idx_theta_i * m_size.y() * m_size.z();
        UInt32 offset_theta_i_marg = idx_theta_i * m_size.z();

        Point2f sample(direction_sample);

        // Avoid degeneracies on the domain boundary
        sample = dr::clamp(sample, dr::Smallest<Float>, dr::OneMinusEpsilon<Float>);

        // Scale sample theta_o range
        sample.x() *= dr::gather<Float>(m_marg_phi_cdf, offset_theta_i_marg + m_size.z() - 1, active);

        // Sample phi
        UInt32 idx_phi = dr::binary_search<UInt32>(
            0u, m_size.z() - 1, [&](UInt32 idx) {
            return dr::gather<Float>(m_marg_phi_cdf, offset_theta_i_marg + idx, active) < sample.x();
            }
        );

        UInt32 offset_phi = idx_phi * m_size.y();
        Float phi_o = (Float) idx_phi / Float(m_size.z()) * dr::TwoPi<Float> + phi_i;

        // Scale theta_o sample range
        sample.y() *= dr::gather<Float>(m_cond_cdf, offset_theta_i_cond + offset_phi + m_size.y() - 1, active);

        // Sample theta_o
        UInt32 idx_theta_o = dr::binary_search<UInt32>(
            0u, m_size.y() - 1, [&](UInt32 idx) {
                return dr::gather<Float>(
                    m_cond_cdf,
                    idx + offset_theta_i_cond + offset_phi,
                    active
                ) < sample.y();
            }
        );

        // PDF value preparations
        Float cond_cdf_0 = dr::gather<Float>(m_cond_cdf, offset_theta_i_cond + offset_phi + idx_theta_o - 1, active && idx_phi > 0);
        Float cond_cdf_1 = dr::gather<Float>(m_cond_cdf, offset_theta_i_cond + offset_phi + idx_theta_o, active);

        Float theta_o = (Float) idx_theta_o / (Float) m_size.y() * (dr::Pi<Float> / 2.f);

        BSDFSample3f bs = dr::zero<BSDFSample3f>();

        active &= cos_theta_i > 0.f;

        bs.wo = Point3f(
                    dr::sin(theta_o)*dr::cos(phi_o),
                    dr::sin(theta_o)*dr::sin(phi_o),
                    dr::cos(theta_o)
        );
        bs.pdf          = dr::select(cos_theta_i > 0.f, (cond_cdf_1 - cond_cdf_0) * m_normalization, 0.f);
        bs.eta          = 1.f;
        bs.sampled_type = +BSDFFlags::GlossyReflection;
        Spectrum value = eval_rpv(si, bs.wo, active);

        return { bs, dr::select(active && bs.pdf > 0.f,
                            depolarizer<Spectrum>(value), 0.f) };
    }

    Spectrum eval_rpv(const SurfaceInteraction3f &si,
                      const Vector3f &wo,
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
             F * (1.f + (1.f - rho_c) / (1 + G))) * dr::InvPi<Float>;;
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

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f &si_,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        //Normalize si.wi
        SurfaceInteraction3f si = si_;
        si.wi = dr::normalize(si.wi);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_theta_o = Frame3f::cos_theta(wo);
        Float cos_phi_i = Frame3f::cos_phi(si.wi);
        Float cos_phi_o = Frame3f::cos_phi(wo);
        Float theta_i = dr::acos(cos_theta_i);
        Float theta_o = dr::acos(cos_theta_o);
        Float phi_i = dr::acos(cos_phi_i);
        Float phi_o = dr::acos(cos_phi_o);
        Float phi = phi_o - phi_i;
        phi = dr::select(phi < 0, phi + dr::TwoPi<Float>, phi);

        UInt32 idx_theta_i = dr::floor2int<UInt32>(theta_i / (dr::Pi<Float> / 2.f) * Float(m_size.x()));
        UInt32 idx_theta_o = dr::floor2int<UInt32>(theta_o / (dr::Pi<Float> / 2.f) * Float(m_size.y()));
        UInt32 idx_phi = dr::floor2int<UInt32>(phi / dr::TwoPi<Float> * Float(m_size.z()));

        // Avoid out of range indices for backside directions
        idx_theta_i = dr::clamp(idx_theta_i, 0u, m_size.x() -1);
        idx_theta_o = dr::clamp(idx_theta_o, 0u, m_size.y() -1);

        UInt32 offset_theta_i = idx_theta_i * m_size.y() * m_size.z();
        UInt32 offset_phi = idx_phi * m_size.y();

        Float cond_cdf_0 = dr::gather<Float>(m_cond_cdf, offset_theta_i + offset_phi + idx_theta_o - 1, active && idx_phi > 0);
        Float cond_cdf_1 = dr::gather<Float>(m_cond_cdf, offset_theta_i + offset_phi + idx_theta_o, active);

        Float pdf = (cond_cdf_1 - cond_cdf_0) * m_normalization;
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
    std::unique_ptr<Float[]> m_data;
    Float m_normalization, m_inv_normalization;
    ScalarPoint3u m_size;

    /// Marginal and conditional PDFs
    FloatStorage m_marg_phi_cdf;
    FloatStorage m_cond_cdf;
};

MI_IMPLEMENT_CLASS_VARIANT(RPV, BSDF)
MI_EXPORT_PLUGIN(RPV, "Rahman-Pinty-Verstraete BSDF")
NAMESPACE_END(mitsuba)
