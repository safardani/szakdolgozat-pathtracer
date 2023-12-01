#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>
#include <optix_device.h>

#include <sutil/vec_math.h>
#include <random.h>

#include <stdio.h>

#define RANDFLOAT3 make_float3(rnd(seed), rnd(seed), rnd(seed))
constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;

// Declare a constant Params structure that will be filled in by the host (CPU) before launch,
// and can be accessed by all the device (GPU) kernels.
extern "C" {
    __constant__ Params params;
}

// Helper class for constructing an orthonormal basis given a normal vector.
struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


// This utility function performs the actual ray tracing from the given origin in the given direction.
// If the ray intersects an object, the payload (prd) contains the shading information which is set in closest-hit or miss program.
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,          // The traversable handle representing the scene to trace against
    float3                 ray_origin,      // The origin of the ray
    float3                 ray_direction,   // The direction of the ray
    float                  tmin,            // The minimum distance along the ray to check for intersections
    float                  tmax,            // The maximum distance along the ray to check for intersections
    Payload* prd                             // The payload registered with the ray
)
{
    // Convert the payload data to 32-bit float values that can be used in the optixTrace call
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;

    u0 = __float_as_uint(prd->attenuation.x);
    u1 = __float_as_uint(prd->attenuation.y);
    u2 = __float_as_uint(prd->attenuation.z);
    u3 = prd->seed;
    u17 = prd->depth;

    // Perform the trace call which will call into the intersect, any-hit and closest-hit programs
    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                        // SBT offset
        1,           // SBT stride
        0,                        // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);
    optixReorder(
        // Application specific coherence hints could be passed in here
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

    prd->attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd->seed = u3;
    
    prd->emitted = make_float3(__uint_as_float(u4), __uint_as_float(u5), __uint_as_float(u6));
    prd->radiance = make_float3(__uint_as_float(u7), __uint_as_float(u8), __uint_as_float(u9));
    prd->origin = make_float3(__uint_as_float(u10), __uint_as_float(u11), __uint_as_float(u12));
    prd->direction = make_float3(__uint_as_float(u13), __uint_as_float(u14), __uint_as_float(u15));
    prd->done = u16;
    prd->depth = u17;
}


// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        1,                         // SBT stride
        0                          // missSBTIndex
    );
    return optixHitObjectIsHit();
}


// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ Payload getPayloadCH()
{
    Payload p = {};

    p.attenuation.x = __uint_as_float(optixGetPayload_0());
    p.attenuation.y = __uint_as_float(optixGetPayload_1());
    p.attenuation.z = __uint_as_float(optixGetPayload_2());
    p.seed = optixGetPayload_3();
    p.depth = optixGetPayload_17();

    return p;
}

// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ Payload getPayloadMiss()
{
    Payload p = {};

    return p;
}


// A helper function to set the payload for the current ray
static __forceinline__ __device__ void setPayloadCH(Payload p)
{
    optixSetPayload_0(__float_as_uint(p.attenuation.x));
    optixSetPayload_1(__float_as_uint(p.attenuation.y));
    optixSetPayload_2(__float_as_uint(p.attenuation.z));
    
    optixSetPayload_3(p.seed);
    
    optixSetPayload_4(__float_as_uint(p.emitted.x));
    optixSetPayload_5(__float_as_uint(p.emitted.y));
    optixSetPayload_6(__float_as_uint(p.emitted.z));
    
    optixSetPayload_7(__float_as_uint(p.radiance.x));
    optixSetPayload_8(__float_as_uint(p.radiance.y));
    optixSetPayload_9(__float_as_uint(p.radiance.z));
    
    optixSetPayload_10(__float_as_uint(p.origin.x));
    optixSetPayload_11(__float_as_uint(p.origin.y));
    optixSetPayload_12(__float_as_uint(p.origin.z));
    
    optixSetPayload_13(__float_as_uint(p.direction.x));
    optixSetPayload_14(__float_as_uint(p.direction.y));
    optixSetPayload_15(__float_as_uint(p.direction.z));
    
    optixSetPayload_16(p.done);
    optixSetPayload_17(p.depth);
}


// A helper function to set the payload for the current ray
static __forceinline__ __device__ void setPayloadMiss(Payload p)
{
    optixSetPayload_4(__float_as_uint(p.emitted.x));
    optixSetPayload_5(__float_as_uint(p.emitted.y));
    optixSetPayload_6(__float_as_uint(p.emitted.z));
    
    optixSetPayload_7(__float_as_uint(p.radiance.x));
    optixSetPayload_8(__float_as_uint(p.radiance.y));
    optixSetPayload_9(__float_as_uint(p.radiance.z));
    
    optixSetPayload_16(p.done);
}

// Helper function to create an orthonormal basis given a normal vector
static __forceinline__ __device__ void createOrthogonalSystem(const float3& w, float3& u, float3& v) {
    if (fabs(w.x) > fabs(w.y))
        u = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    else
        u = normalize(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    v = cross(w, u);
}

// A helper function to generate a random 3D vector that is distributed as a cosine-weighted hemisphere around the z-axis
static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// A helper function to generate a random 3D vector that is inside the unit sphere (i.e. length < 1.0f)
static __forceinline__ __device__ float3 random_in_unit_sphere(unsigned int seed) {
    float3 p;
    do {
        p = 2.0f * RANDFLOAT3 - make_float3(1, 1, 1);
    } while (p.x * p.x + p.y * p.y + p.z * p.z >= 1.0f);
    return p;
}

// Filmic tonemap operator
static __forceinline__ __device__ float3 tonemap(float3 x)
{
    // Coefficients of a rational polynomial fit to the ACES filmic tone mapping curve.
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

static __forceinline__ __device__ float3 defocus_disk_sample(float3 u, float3 v, unsigned int seed) {
    // Returns a random point in the camera defocus disk.

    const float r = sqrt(rnd(seed));  // Uniformly distributed radius
    const float theta = 2.0f * M_PI * rnd(seed);  // Uniformly distributed angle

    float blurriness = 0.01f;

    // Unit disk x, y coordinates
    const float x = blurriness * sqrt(r) * cosf(theta);
    const float y = blurriness * sqrt(r) * sinf(theta);

    float3 dir = (x * u + y * v);

    return dir;
}

// The ray generation program. This is called once per pixel, and its job is to generate the primary rays.
extern "C" __global__ void __raygen__rg()
{
    // Get a pointer to the ray generation data stored in the Shader Binding Table
    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    
    // Get the camera basis vectors and eye position from the ray generation data.
    const float3      eye = rtData->cam_eye;
    const float3      U = rtData->camera_u;
    const float3      V = rtData->camera_v;
    const float3      W = rtData->camera_w;

    float disk_radius = 2.4f;

    // Compute the launch index, which corresponds to the pixel indices in the image
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    unsigned int seed = tea<4>(idx.y, idx.x);

    // Set an initial color payload for the ray which might be modified by the closest-hit or miss program
    float3       payload_rgb = make_float3(0.0f);

    // Sample the pixel multiple times and average the results
    int sample_batch_count = 600;
    for (size_t i = 0; i < sample_batch_count; i++)
    {
        // Generate a random subpixel offset for anti-aliasing
        float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        float focus_distance = 1.0f;

        // Normalized device coordinates (NDC) are in the range [-1, 1] for both x and y
        float2 d = 2.0f * make_float2((idx.x + subpixel_jitter.x) / (dim.x), (idx.y + subpixel_jitter.y) / (dim.y)) - 1.0f;
        
        // Calculate the ray origin and direction for the current pixel
        float3 origin = defocus_disk_sample(U,V, seed); // rtData->cam_eye;
        float3 target = focus_distance * (d.x * U + d.y * V + W);
        float3 direction = normalize(target - origin);
        origin += rtData->cam_eye;

        // Reset the payload data for this ray (every iteration of the loop)
        Payload payload;
        payload.attenuation = make_float3(1.0f);
        payload.seed = seed;
        payload.depth = 0.f;

        int max_depth = 20;
        // Trace the ray into the scene
        for (;;)
        {
            traceRadiance(
                params.handle,
                origin,
                direction,
                0.01f,  // tmin
                1e16f,  // tmax
                &payload);

            // Accumulate the ray's contribution to the pixel color
            payload_rgb += payload.emitted;
            payload_rgb += payload.radiance * payload.attenuation;

            // Russian roulette: try to keep path weights equal to one by randomly terminating paths.
            const float p = dot(payload.attenuation, make_float3(0.30f, 0.59f, 0.11f));
            const bool done = payload.done || rnd(seed) > p;
            if (done)
                break;
            payload.attenuation /= p;

            // Update ray origin and direction for the next path segment
            origin = payload.origin;
            direction = payload.direction;

            ++payload.depth;
        }
    }

    // Exposure compensation value
    float exposure = 0.0f;

    // Apply exposure before tonemapping
    payload_rgb = payload_rgb / sample_batch_count * exp2(exposure); // Incorporate exposure

    // Apply filmic tonemapping to the HDR values
    payload_rgb = tonemap(payload_rgb);

    // Scale to [0, 1] range
    payload_rgb = clamp(payload_rgb, 0.0f, 1.0f);

    // Optional: Apply simple gamma correction post tonemapping
    float gamma = 2.2f;
    payload_rgb = make_float3(
        powf(payload_rgb.x, 1.0f / gamma),
        powf(payload_rgb.y, 1.0f / gamma),
        powf(payload_rgb.z, 1.0f / gamma));

    // Write the tonemapped and gamma-corrected pixel color to the image buffer
    params.image[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
}

// Fresnel-Schlick implementation for specular reflection
static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, const float3& F0)
{
    return F0 + (make_float3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

// The miss program. This is called for any ray that does not hit geometry.
extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    // Retrieve the current MissData from the SBT
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    Payload prd = getPayloadMiss();

    float3 ray_dir = optixGetWorldRayDirection();
    float3 sunlight_direction = normalize(make_float3(0.9f, 1.0f, 3.0f));

    // Set the ray's payload to the miss color (in this case black)
    if (length(ray_dir - sunlight_direction) < 0.1f)
		prd.radiance = make_float3(15.0f);
	else
		prd.radiance = make_float3(rt_data->r, rt_data->g, rt_data->b);

    prd.emitted = make_float3(0.f);
    prd.done = true;

    setPayloadMiss(prd);
}

// The closest hit program. This is called when a ray hits the closest geometry.
extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    // Retrieve the current HitGroupData from the SBT
    HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Retrieve the primitive index and barycentrics for the current hit
    const unsigned int  prim_idx = optixGetPrimitiveIndex();
    const float3        ray_dir = optixGetWorldRayDirection();
    const float3        ray_orig = optixGetWorldRayOrigin();
    float               t_hit = optixGetRayTmax();
    const int           vert_idx_offset = prim_idx * 3;

    // Other information such as primitive index, traversable handle and SBT GAS index
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    // Define and retrieve the sphere's data (center and radius)
    float4 q;
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);

    // Compute the intersection point in world space
    float3 intersect_point = ray_orig + t_hit * ray_dir;
    // Transform the intersection point from world space to object space
    float3 localcoords_intersect_location = optixTransformPointFromWorldToObjectSpace(intersect_point);
    // Determine the object space normal, and then transform it back to world space
    float3 localcoords_obj_normal = (localcoords_intersect_location - make_float3(q)) / q.w;
    // Normalize the normal vector after transforming it to world space
    float3 normal_intersect = normalize(optixTransformNormalFromObjectToWorldSpace(localcoords_obj_normal));

    // Read the existing payload. This payload was set by the raygen program.
    Payload p = getPayloadCH();

    // If this is the first bounce, set the emitted color to the object's emission color
    if (p.depth == 0)
        p.emitted = hit_group_data->emission_color;
    else
        p.emitted = make_float3(0.0f);

    // Calculate the roughness, and square it to make it more pronounced
    float roughness = hit_group_data->roughness;
    roughness *= roughness;

    bool metallic = hit_group_data->metallic;
    bool transparent = hit_group_data->transparent;
    
    // Compute the Fresnel-Schlick term
    float3 F0 = (1 - make_float3(1.45f)) / (1 + make_float3(1.45f)); // Assume non-metallic
    F0 = F0 * F0;
    float3 F = fresnelSchlick(fmaxf(dot(normal_intersect, -ray_dir), 0.0f), F0);

    if (transparent) {
        float3 refracted_ray;

		// calcualte if we are entering or exiting the sphere
		bool entering = dot(ray_dir, normal_intersect) < 0.0f;
		
        float sphere_ior = 1.45f;
		float refraction_ratio = !entering ? 1.0 / sphere_ior : sphere_ior;
        bool refracted = refract(refracted_ray, ray_dir, normal_intersect, refraction_ratio);
        float3 normal = entering ? normal_intersect : -normal_intersect;

        unsigned int seed = p.seed;
        if ((rnd(seed) > F.x || !entering) && refracted)
        {
            p.direction = normalize(refracted_ray + roughness * normalize(random_in_unit_sphere(seed)));
        } else {
            p.direction = normalize(reflect(ray_dir, normal) + roughness * normalize(random_in_unit_sphere(seed)));
        }
		
		p.origin = intersect_point;
		
        // print depth
        // printf("Depth: %d\n", p.depth);
		
		setPayloadCH(p);
        return;
    }

    // Use the provided color and other properties from the hit group data
    const float3 diffuse_albedo = hit_group_data->diffuse_color;
    const float3 specular_albedo = hit_group_data->specular;

    // Compute the Fresnel-Schlick term
    F0 = metallic ? make_float3(0.8f) : make_float3(0.04f); // Assume non-metallic
    F = fresnelSchlick(fmaxf(dot(normal_intersect, -ray_dir), 0.0f), F0);
		
    unsigned int seed = p.seed;
    {
        /*// For cosine-weighted hemisphere sampling, TODO reimplement later
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        // Generate a random direction for diffuse reflection
        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(normal_intersect);
        onb.inverse_transform(w_in);

        p.direction = w_in;
        */

        // Combine the specular and diffuse components
        float3 specular_component = F * specular_albedo;
        float3 diffuse_component = (1.0f - F) * diffuse_albedo;

        // Combine the specular and diffuse components by adding them together
        float3 material_response = metallic ? specular_component : diffuse_component + specular_component;

        // Calculate the perfect specular reflection direction and the diffuse reflection direction
        float3 specular_dir = -normalize(reflect(-ray_dir, normal_intersect));
        float3 diffuse_dir = normalize(normal_intersect + random_in_unit_sphere(p.seed));

        float3 new_dir;
        unsigned int seed = p.seed;
        float random = rnd(seed);

        // Calculate the probability of sampling the diffuse component
        if (random < F.x || metallic) {
            // Specular component
            new_dir = normalize((1.0f - roughness) * specular_dir + roughness * diffuse_dir);
        }
        else {
            // Diffuse component
            new_dir = diffuse_dir;
        }
        float3 cur_attenuation = material_response * p.attenuation;

        // Set the payload data for the next iteration of the ray
        p.direction = new_dir;
        p.origin = intersect_point;

        // Calculate the diffuse component of the material
        p.attenuation = cur_attenuation;
    }

    p.seed = seed;

    // Define the sunlight properties
    float3 sunlight_direction = normalize(make_float3(0.9f, 1.0f, 3.0f));
    float3 sunlight_center_direction = sunlight_direction;
    float3 sunlight_emission = make_float3(15.0f);
    float sunlight_spread = 0.1f;

    // Sample the sun's direction
    const float r = sqrt(rnd(seed));  // Uniformly distributed radius
    const float theta = 2.0f * M_PI * rnd(seed);  // Uniformly distributed angle

    // Unit disk x, y coordinates
    const float x = sqrt(r) * cosf(theta);
    const float y = sqrt(r) * sinf(theta);

    // Construct an orthonormal basis with w as sunlight_direction
    float3 w = sunlight_direction;
    float3 u, v;
    createOrthogonalSystem(w, u, v);

    // Sample the sun's direction
    sunlight_direction = normalize(sunlight_direction + sunlight_spread * (x * u + y * v));

    // Calculate properties of light sample (for area based pdf)
    const float  nDl = dot(normal_intersect, sunlight_direction);

    // Calculate the weight of the sunlight sample
    float weight = 0.0f;
    if (nDl > 0.0f)
    {
        const bool occluded =
            traceOcclusion(
                params.handle,
                intersect_point,
                sunlight_direction,
                0.01f,           // tmin
                1e16f);  // tmax

        // If the light is not occluded, calculate the weight
        if (!occluded)
            weight = metallic ? 0.0f : nDl;

        // Calculate reflection direction
        float3 reflection_direction = reflect(-ray_dir, normal_intersect);
        reflection_direction = -normalize(reflection_direction + roughness * normalize(random_in_unit_sphere(p.seed)));

        // Calculate if we need to show the specular highlight
        if (length(reflection_direction - sunlight_center_direction) < sunlight_spread)
        {
            p.radiance = F * sunlight_emission * nDl; // TODO do we need to multiply by nDl?
        }

    }

    // Calculate the radiance of the sunlight sample
    p.radiance += sunlight_emission * weight;
    p.done = false;

    setPayloadCH(p);
}