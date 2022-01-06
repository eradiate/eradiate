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
.. _volume-gridvolume_spherical:

Grid-based volume data source in spherical coordinates(:monosp:`gridvolume_spherical`)
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
class GridVolumeSpherical final : public Volume<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Volume, m_to_local, update_bbox)
    MTS_IMPORT_TYPES(Volume, VolumeGrid)

    GridVolumeSpherical(const Properties &props) : Base(props) {
        m_gridvol = props.volume<Volume>("gridvolume", 1.f);
        m_inner_bsphere_radius_relative = props.get<ScalarFloat>("inner_bsphere_radius", 0.f);
    }

    UnpolarizedSpectrum eval(const Interaction3f &it, Mask active) const override {
        auto p = m_to_local * it.p;

        // transform from the [0:1, 0:1, 0:1] cube to a  [-1:1, -1:1, -1:1] cube
        ScalarTransform4f to_center = ScalarTransform4f::translate(ScalarPoint3f(-1, -1, -1)) * ScalarTransform4f::scale(2.f);
        p = to_center * p;

        Point3f p_spherical = Point3f(
            (ek::norm(p) - m_inner_bsphere_radius_relative) / (1.f - m_inner_bsphere_radius_relative),
            ek::acos(p.z() / ek::norm(p)) / ek::Pi<ScalarFloat>,
            ek::atan2(p.y(), p.x()) / ek::Pi<ScalarFloat> + 1.f
        );

        Interaction3f it_spherical = it;
        it_spherical.p = p_spherical;

        Mask within_spherical_shell = (m_inner_bsphere_radius_relative <= ek::norm(p) && ek::norm(p) <= 1.0);

        return ek::select(within_spherical_shell, m_gridvol->eval(it_spherical, active), 0.f);
    }

    ScalarFloat max() const override { return m_gridvol->max(); }

    ScalarVector3i resolution() const override { return m_gridvol->resolution(); };

    MTS_DECLARE_CLASS()

protected:
    ScalarFloat m_inner_bsphere_radius_relative;
    ref<Volume> m_gridvol;
};

MTS_IMPLEMENT_CLASS_VARIANT(GridVolumeSpherical, Volume)
MTS_EXPORT_PLUGIN(GridVolumeSpherical, "GridVolumeSpherical texture")

NAMESPACE_END(mitsuba)
