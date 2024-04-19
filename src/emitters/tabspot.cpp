#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-spot:

Tabulated Spot light source (:monosp:`spot`)
------------------------------------

.. pluginparameters::

 * - intensity
   - |spectrum|
   - Specifies the maximum radiant intensity at the center in units of power per unit steradian. (Default: 1).
     This cannot be spatially varying (e.g. have bitmap as type).
   - |exposed|, |differentiable|

 * - falloff_table
   - |float|
   - A comma-separated list of falloff table values (from 1 centre to 0) parametrized by the
     the scattering from the central ray, i.e, theta_0, val_0, .. , theta_i, val_i, theta_N, val_N.
   - |exposed|, |differentiable|

 * - texture
   - |texture|
   - An optional texture to be projected along the spot light. This must be spatially varying (e.g. have bitmap as type).
   - |exposed|, |differentiable|

 * - to_world
   - |transform|
   - Specifies an optional emitter-to-world transformation.  (Default: none, i.e. emitter space = world space)
   - |exposed|

This plugin provides a spot light with a linear falloff. In its local coordinate system, the spot light is
positioned at the origin and points along the positive Z direction. It can be conveniently reoriented
using the lookat tag, e.g.:

.. tabs::
    .. code-tab:: xml
        :name: spot-light

        <emitter type="spot">
            <transform name="to_world">
                <!-- Orient the light so that points from (1, 1, 1) towards (1, 2, 1) -->
                <lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
            </transform>
            <rgb name="intensity" value="1.0"/>
        </emitter>

    .. code-tab:: python

        'type': 'spot',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[1, 1, 1],
            target=[1, 2, 1],
            up=[0, 0, 1]
        ),
        'intensity': {
            'type': 'spectrum',
            'value': 1.0,
        }

The intensity linearly ramps up from cutoff_angle to beam_width (both specified in degrees),
after which it remains at the maximum value. A projection texture may optionally be supplied.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/emitter_spot_no_texture.jpg
   :caption: Two spot lights with different colors and no texture specified.
.. subfigure:: ../../resources/data/docs/images/render/emitter_spot_texture.jpg
   :caption: A spot light with a texture specified.
.. subfigend::
   :label: fig-spot-light

 */

template <typename Float, typename Spectrum>
class TabSpotLight final : public Emitter<Float, Spectrum> {
using FloatStorage             = DynamicBuffer<Float>;
using Size                     = uint32_t;
//using Index                    = dr::uint32_array_t<Float>;
using Index                    = uint32_t;

public:
    MI_IMPORT_BASE(Emitter, m_flags, m_medium, m_to_world)
    MI_IMPORT_TYPES(Scene, Texture)

    TabSpotLight(const Properties &props) : Base(props) {

        m_flags = +EmitterFlags::DeltaPosition;
        m_intensity = props.texture_d65<Texture>("intensity", 1.f);
        m_texture = props.texture_d65<Texture>("texture", 1.f);

        if (m_intensity->is_spatially_varying())
            Throw("The parameter 'intensity' cannot be spatially varying (e.g. bitmap type)!");

        if (props.has_property("texture")) {
            if (!m_texture->is_spatially_varying())
                Throw("The parameter 'texture' must be spatially varying (e.g. bitmap type)!");
            m_flags |= +EmitterFlags::SpatiallyVarying;
        }
        dr::set_attr(this, "flags", m_flags);

       if (props.type("falloff_table") == Properties::Type::String) {
            std::vector<std::string> values_str =
                string::tokenize(props.string("falloff_table"), " ,");
            std::vector<ScalarFloat> nodes, data;
            nodes.reserve(values_str.size()/2);      // angles
            data.reserve(values_str.size()/2);      // values

            bool first = true;
            for (const auto &s : values_str) {

                if (first) {
                       try {
                        nodes.push_back((ScalarFloat) std::stod(s));
                    } catch (...) {
                        Throw("Could not parse floating point value '%s'", s);
                    }
                    first = false;
                }

                else {
                    try {
                        data.push_back((ScalarFloat) std::stod(s));
                    } catch (...) {
                        Throw("Could not parse floating point value '%s'", s);
                    }
                    first = true;
                }
            }

            // deg -> rad
            m_tab_angles = dr::deg_to_rad( dr::load<FloatStorage>(nodes.data(), values_str.size()/2) );
            m_tab_values = dr::load<FloatStorage>(data.data(), values_str.size()/2);
            // assume the last entry to be 0.
            m_cutoff_angle = m_tab_angles[(values_str.size()/2)-1];
            m_cos_cutoff_angle = dr::cos(m_cutoff_angle);
            m_uv_factor = dr::tan(m_cutoff_angle);
        }

        else {
            Throw("'falloff_table' must be a string of the form");
        }

        dr::make_opaque(m_tab_angles, m_tab_values, m_cutoff_angle, m_uv_factor,
                        m_cos_cutoff_angle);
        
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("intensity",        m_intensity.get(), +ParamFlags::Differentiable);
        callback->put_object("texture",          m_texture.get(),   +ParamFlags::Differentiable);
        callback->put_parameter("falloff_angles",m_tab_angles,      +ParamFlags::NonDifferentiable);
        callback->put_parameter("falloff_values",m_tab_values,      +ParamFlags::Differentiable); //| ParamFlags::Discontinuous);
        callback->put_parameter("to_world",     *m_to_world.ptr(),  +ParamFlags::NonDifferentiable);
    }

    /**
     * Computes the UV coordinates corresponding to a direction in the local frame.
     */
    Point2f direction_to_uv(const Vector3f &local_dir) const {
        return Point2f(
            0.5f + 0.5f * local_dir.x() / (local_dir.z() * m_uv_factor),
            0.5f + 0.5f * local_dir.y() / (local_dir.z() * m_uv_factor)
        );
    }

    /**
     * Returns a factor in [0, 1] accounting for the falloff profile of
     * the spot emitter in direction `d`.
     *
     * Does not include the emitted radiance in that direction.
     */
    Float falloff_curve(const Vector3f &d, Mask /*active*/) const {
        using Index = dr::uint32_array_t<Float>;

        Vector3f local_dir = dr::normalize(d);
        Float theta    = dr::acos(local_dir.z());
        
        /*
        // find interval in ordered set
        UInt32 i = math::find_interval<UInt32>(
                    (uint32_t) m_tab_angles.size(),
                    [&](UInt32 idx) DRJIT_INLINE_LAMBDA {
                        return dr::gather<Float>(m_tab_angles, idx) <= theta;
                    });
        */
        Index i = math::find_interval<Index>(
                    (uint32_t) m_tab_angles.size(),
                    [&](Index idx) DRJIT_INLINE_LAMBDA {
                        return dr::gather<Float>(m_tab_angles, idx) <= theta;
                    });

        // linerarly interpolate RID
        Float weight = (theta - dr::gather<Float>(m_tab_angles, i)) / (dr::gather<Float>(m_tab_angles, i+1) - dr::gather<Float>(m_tab_angles, i));
        Float falloff =  dr::lerp(dr::gather<Float>(m_tab_values, i), dr::gather<Float>(m_tab_values, i+1), weight);
        //std::cout << "min falloff " <<  dr::min(falloff) << " ++ max falloff " <<  dr::max(falloff) << std::endl;
        return dr::select(theta <= m_cutoff_angle, falloff, 0.f);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*dir_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample directional component
        Vector3f local_dir = warp::square_to_uniform_cone(spatial_sample, (Float) m_cos_cutoff_angle);
        Float pdf_dir = warp::square_to_uniform_cone_pdf(local_dir, (Float) m_cos_cutoff_angle);

        // 2. Sample spectrum
        auto si = dr::zeros<SurfaceInteraction3f>();
        si.time = time;
        si.p    = m_to_world.value().translation();
        si.uv   = direction_to_uv(local_dir);
        auto [wavelengths, spec_weight] =
            sample_wavelengths(si, wavelength_sample, active);

        Float falloff = falloff_curve(local_dir, active);

        return { Ray3f(si.p, m_to_world.value() * local_dir, time, wavelengths),
                 depolarizer<Spectrum>(spec_weight * falloff / pdf_dir) };
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &/*sample*/,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        DirectionSample3f ds;
        ds.p        = m_to_world.value().translation();
        ds.n        = 0.f;
        ds.uv       = 0.f;
        ds.pdf      = 1.f;
        ds.time     = it.time;
        ds.delta    = true;
        ds.emitter  = this;
        ds.d        = ds.p - it.p;
        ds.dist     = dr::norm(ds.d);
        Float inv_dist = dr::rcp(ds.dist);
        ds.d        *= inv_dist;
        Vector3f local_d = m_to_world.value().inverse() * -ds.d;

        // Evaluate emitted radiance & falloff profile
        Float falloff = falloff_curve(local_d, active);
        active &= falloff > 0.f;  // Avoid invalid texture lookups

        SurfaceInteraction3f si      = dr::zeros<SurfaceInteraction3f>();
        si.t                         = 0.f;
        si.time                      = it.time;
        si.wavelengths               = it.wavelengths;
        si.p                         = ds.p;
        UnpolarizedSpectrum radiance = m_intensity->eval(si, active);
        if (m_texture->is_spatially_varying()) {
            si.uv = direction_to_uv(local_d);
            radiance *= m_texture->eval(si, active);
        }

        return { ds, depolarizer<Spectrum>(radiance & active) * (falloff * dr::sqr(inv_dist)) };
    }

    Float pdf_direction(const Interaction3f &,
                        const DirectionSample3f &, Mask) const override {
        return 0.f;
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f & /*sample*/,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);

        Vector3f center_dir = m_to_world.value() * ScalarVector3f(0.f, 0.f, 1.f);
        PositionSample3f ps(
            /* position */ m_to_world.value().translation(), center_dir,
            /*uv*/ Point2f(0.5f), time, /*pdf*/ 1.f, /*delta*/ true
        );
        return { ps, Float(1.f) };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        Wavelength wav;
        Spectrum weight;

        if (m_texture->is_spatially_varying()) {
            std::tie(wav, weight) = m_texture->sample_spectrum(
                si, math::sample_shifted<Wavelength>(sample), active);

            SurfaceInteraction3f si2 = si;
            si2.wavelengths = wav;
            weight *= m_intensity->eval(si2, active);
        } else {
            std::tie(wav, weight) = m_intensity->sample_spectrum(
                si, math::sample_shifted<Wavelength>(sample), active);
        }

        return { wav, weight };
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override {
        return 0.f;
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "TabSpotLight[" << std::endl
            << "  to_world = " << string::indent(m_to_world) << "," << std::endl
            << "  intensity = " << m_intensity << "," << std::endl
            << "  tab_angles=" << m_tab_angles << "," << std::endl
            << "  tab_values=" << m_tab_values << "," << std::endl
            << "  cutoff_angle = " << m_cutoff_angle << "," << std::endl
            << "  texture = " << (m_texture ? string::indent(m_texture) : "")
            << "  medium = " << (m_medium ? string::indent(m_medium) : "")
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_intensity;
    ref<Texture> m_texture;
    FloatStorage m_tab_angles;
    FloatStorage m_tab_values;
    //Float m_tab_angles, m_tab_values
    Float m_cutoff_angle, m_cos_cutoff_angle, m_uv_factor;
    //Float m_cos_beam_width, m_cos_cutoff_angle, m_inv_transition_width;
};


MI_IMPLEMENT_CLASS_VARIANT(TabSpotLight, Emitter)
MI_EXPORT_PLUGIN(TabSpotLight, "Tabulated Spot emitter")
NAMESPACE_END(mitsuba)
