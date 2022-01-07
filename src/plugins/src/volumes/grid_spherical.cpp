#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>

#include <mitsuba/render/srgb.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/volumegrid.h>
#include <enoki/dynamic.h>
#include <enoki/tensor.h>

NAMESPACE_BEGIN(mitsuba)

enum class FilterType { Nearest, Trilinear };
enum class WrapMode { Repeat, Mirror, Clamp };

/**!
.. _volume-sphericalgridvolume:

Grid-based volume data source in spherical coordinates(:monosp:`sphericalgridvolume`)
----------------------------------------------------------

.. pluginparameters::

 * - gridvolume
   - |gridvolume|
   - Nested gridvolume object that holds the actual data organized in spherical coordinates
     Method calls are pre processed if necessary and then forwarded to this object.

 * - inner_bsphere_radius
   - |float|
   - Specifies the relative radius of the inner bounding sphere of the spherical shell
     that delimits this volume. This must be within [0, 1.0]. 

This class implements a gridvolume in spherical coordinates. To achieve that, it holds a nested
gridvolume object and preprocesses method calls accordingly before forwarding them to the nested
object. In the :monosp:`eval()` method, the sampled point is converted to spherical coordinates,
relative to the center of the sphere and converted such that :math:`r`, :math:`\theta` and :math:`\phi`
cover the [0,1] range.

*/

template <typename Float, typename Spectrum>
class SphericalGridVolume final : public Volume<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Volume, m_to_local, m_bbox)
    MTS_IMPORT_TYPES(Volume, VolumeGrid)

    SphericalGridVolume(const Properties &props) : Base(props) {
        m_gridvol = props.volume<Volume>("gridvolume", 1.f);

        m_rmin = props.get<ScalarFloat>("rmin", 0.f);
        m_rmax = props.get<ScalarFloat>("rmax", 1.f);
        m_rmin_rel = m_rmin / m_rmax;

        m_fillmin = props.get<ScalarFloat>("fillmin", 0.f);
        m_fillmax = props.get<ScalarFloat>("fillmax", 0.f);

        m_to_local = props.get<ScalarTransform4f>("to_world", ScalarTransform4f()).inverse();
        update_bbox_sphere();
    }

    UnpolarizedSpectrum eval(const Interaction3f &it, Mask active) const override {
        Point3f p = m_to_local * it.p;
        Float p_magnitude = ek::norm(p);

        Point3f p_spherical = Point3f(
            (p_magnitude - m_rmin) / (m_rmax - m_rmin),
            ek::acos(p.z() / p_magnitude) * ek::InvPi<ScalarFloat>,
            ek::atan2(p.y(), p.x()) * ek::InvPi<ScalarFloat> + 1.f
        );

        Interaction3f it_spherical = it;
        it_spherical.p = p_spherical;

        return ek::select(p_magnitude < m_rmin,
            m_fillmin,
            ek::select(p_magnitude > m_rmax,
                m_fillmax,
                m_gridvol->eval(it_spherical, active)
            )
        );
    }

    ScalarFloat max() const override { return m_gridvol->max(); }

    ScalarVector3i resolution() const override { return m_gridvol->resolution(); };

    MTS_DECLARE_CLASS()

protected:
    ScalarFloat m_center, m_rmin, m_rmax, m_rmin_rel, m_fillmin, m_fillmax;
    ref<Volume> m_gridvol;

    void update_bbox_sphere() {
        ScalarTransform4f to_world = m_to_local.inverse();
        ScalarPoint3f a = to_world * ScalarPoint3f(-1.f, -1.f, -1.f);
        ScalarPoint3f b = to_world * ScalarPoint3f(1.f, 1.f, 1.f);
        m_bbox = ScalarBoundingBox3f(a, b);
    }
};

MTS_IMPLEMENT_CLASS_VARIANT(SphericalGridVolume, Volume)
MTS_EXPORT_PLUGIN(SphericalGridVolume, "SphericalGridVolume texture")

NAMESPACE_END(mitsuba)
