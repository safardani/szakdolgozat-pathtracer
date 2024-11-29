#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>
#include <optix_device.h>

#include <sutil/vec_math.h>
#include <random.h>

#include <stdio.h>

#define RANDFLOAT3 make_float3(myrnd(seed), myrnd(seed), myrnd(seed))
constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;

// Declare a constant Params structure that will be filled in by the host (CPU) before launch,
// and can be accessed by all the device (GPU) kernels.
extern "C" {
    __constant__ Params params;
}

static __forceinline__ __device__ float pcg_hash(unsigned int input) {
    // pcg hash
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float myrnd(unsigned int& seed) {
	seed = pcg_hash(seed);
	return (float)seed / UINT_MAX;
}

// Helper class for constructing an orthonormal basis given a normal vector.
struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normalize(normal);

        // Choose an arbitrary vector that is not parallel to n
        float3 up = fabsf(m_normal.y) < 0.9999f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);

		unsigned int seed = normal.x * 43758 + normal.y * 35413 + normal.z * 12345;

        m_tangent = normalize(cross(up, m_normal));
        m_binormal = normalize(cross(m_normal, m_tangent));
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_normal + p.z * m_binormal;
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
    Payload* prd                            // The payload registered with the ray
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
	p.alt_seed = optixGetPayload_18();

    return p;
}

// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ Payload getPayloadMiss()
{
    Payload p = {};
	
    // p.radiance.x = __uint_as_float(optixGetPayload_7());
    // p.radiance.y = __uint_as_float(optixGetPayload_8());
    // p.radiance.z = __uint_as_float(optixGetPayload_9());

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

	optixSetPayload_18(p.alt_seed);
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
static __forceinline__ __device__ float3 random_in_unit_sphere(unsigned int &seed) {
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

    const float r = sqrt(myrnd(seed));  // Uniformly distributed radius
    const float theta = 2.0f * M_PI * myrnd(seed);  // Uniformly distributed angle

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
    const float3      eye = params.eye;
    const float3      U = params.U;
    const float3      V = params.V;
    const float3      W = params.W;
    const int    subframe_index = params.subframe_index;

    float disk_radius = 2.4f;

    // Compute the launch index, which corresponds to the pixel indices in the image
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // unsigned int seed = tea<4>(idx.y * params.image_width + idx.x, subframe_index);
	// unsigned int alt_seed = seed ^ 0xAAAAAAAA;

    // TODO clean up
    unsigned int unique_key = idx.y * params.image_width + idx.x + subframe_index * params.image_width * params.image_height;
    unsigned int seed = unique_key; //hash(unique_key);
    unsigned int alt_seed = unique_key + 1; //hash(unique_key + 1);

    // Set an initial color payload for the ray which might be modified by the closest-hit or miss program
    float3       payload_rgb = make_float3(0.0f);

    // Sample the pixel multiple times and average the results
    int sample_batch_count = 10;
    for (size_t i = 0; i < sample_batch_count; i++)
    {
        // Generate a random subpixel offset for anti-aliasing

        float2 subpixel_jitter = make_float2(myrnd(seed), myrnd(seed));
        float focus_distance = 1.0f;

        // Normalized device coordinates (NDC) are in the range [-1, 1] for both x and y
        float2 d = 2.0f * make_float2((idx.x + subpixel_jitter.x) / (dim.x), (idx.y + subpixel_jitter.y) / (dim.y)) - 1.0f;
        
        float3 origin;
        float3 target;
        float3 direction;
		
        if (params.dof) {
            // Calculate the ray origin and direction for the current pixel
            origin = defocus_disk_sample(U,V, seed);
            target = focus_distance * (d.x * U + d.y * V + W);
            direction = normalize(target - origin);
            origin += params.eye;
        } else {
            origin = params.eye;
			direction = normalize(d.x * U + d.y * V + W);
        }

        // Reset the payload data for this ray (every iteration of the loop)
        Payload payload;
        payload.attenuation = make_float3(1.0f);
        payload.seed = seed;
		payload.alt_seed = alt_seed;

		payload.depth = 20; // max_depth;
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
            const bool done = payload.done || myrnd(seed) > p;
            if (done)
                break;
            payload.attenuation /= p;

            // Update ray origin and direction for the next path segment
            origin = payload.origin;
            direction = payload.direction;

            --payload.depth;
        }
    }

    const unsigned int image_index = idx.y * params.image_width + idx.x;
    float3 accum_color = payload_rgb / sample_batch_count;

    if (subframe_index > 0)
    {
        const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);

    // Exposure compensation value
    float exposure = 0.0f;

    // Apply exposure before tonemapping
    payload_rgb = accum_color * exp2(exposure); // Incorporate exposure

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

	params.frame_buffer[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
}

// GGX Normal Distribution Function (NDF)
static __forceinline__ __device__ float D_GGX(float3 n, float3 h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(n, h), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PIf * denom * denom;

    return a2 / denom;
}

// Geometry Function (G) using Schlick-GGX
static __forceinline__ __device__ float G_SchlickGGX(float alpha, float3 n, float3 x)
{
	float numerator = fmaxf(dot(n, x), 0.0f);

	float k = alpha / 2.0f;
	float denominator = fmaxf(dot(n, x), 0.0f) * (1.0f - k) + k;
	denominator = fmaxf(denominator, 0.000001f);

	return numerator / denominator;
}

static __forceinline__ __device__ float G_Smith(float alpha, float3 N, float3 V, float3 L)
{
	return G_SchlickGGX(alpha, N, V) * G_SchlickGGX(alpha, N, L);
}

// Fresnel-Schlick approximation
static __forceinline__ __device__ float3 Fresnel_Schlick(float cosTheta, float3 F0)
{
    cosTheta = clamp(cosTheta, 0.0f, 1.0f); // Ensure valid range
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
		prd.radiance += make_float3(15.0f);
	else
		prd.radiance += make_float3(rt_data->r, rt_data->g, rt_data->b);

    prd.emitted = make_float3(0.f);
    prd.done = true;

    setPayloadMiss(prd);
}

// The closest hit program. This is called when a ray hits the closest geometry.
extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    const unsigned int  sphere_idx = optixGetPrimitiveIndex();
	const float3        ray_dir = optixGetWorldRayDirection(); // direction that the ray is heading in, from the origin
    const float3        ray_orig = optixGetWorldRayOrigin();
	float               t_hit = optixGetRayTmax(); // distance to the hit point

	// get the geometry acceleration structure so we can get the sphere's properties
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

	float4 sphere_props; // stores the 3 center coordinates and the radius
    optixGetSphereData(gas, sphere_idx, sbtGASIndex, 0.f, &sphere_props);

	float3 sphere_center = make_float3(sphere_props.x, sphere_props.y, sphere_props.z);
	float  sphere_radius = sphere_props.w;

	float3 hit_pos = ray_orig + t_hit * ray_dir; // in world space
    float3 localcoords_hit_pos = optixTransformPointFromWorldToObjectSpace(hit_pos);
    float3 normal = normalize(hit_pos - sphere_center); // in world space

    Payload payload = getPayloadCH();
	unsigned int seed = payload.seed;

    float3 specular_albedo = hit_group_data->specular;
	float3 diffuse_albedo = hit_group_data->diffuse_color;
	float3 emission_color = hit_group_data->emission_color;

	float roughness = hit_group_data->roughness; roughness *= roughness;
	float metallicity = hit_group_data->metallic ? 1.0f : 0.0f;
	float transparency = hit_group_data->transparent ? 1.0f : 0.0f;
	float ior = 1.5f;

	if (payload.depth == 0)
        payload.emitted = emission_color; // TODO why are we doing this
    else
        payload.emitted = make_float3(0.0f);

	random_in_unit_sphere(seed); // we need this FOR SOME REASON??? to keep the seed random and avoid artifacts. TODO

	// GGX importance sampling code
    float r1 = myrnd(seed);
    float r2 = myrnd(seed);
    if (roughness < 0.015f) roughness = 0.015f; // Prevent artifacts from division by very small numbers

    float phi = 2.0f * M_PIf * r1;
	float alpha = roughness * roughness;
    float cosTheta = sqrt((1.0f - r2) / (1.0f + (alpha * alpha - 1.0f) * r2));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    float3 half_vec = normalize(make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi)));

	// transform half vector from tangent space to world space
    Onb onb(normal);
    onb.inverse_transform(half_vec);

    float3 light_dir = reflect(ray_dir, half_vec);
    
	float3 F0 = make_float3(fabs((1.0 - ior) / (1.0 + ior))); // 0.04 for dielectrics
    F0 = F0 * F0;
    F0 = lerp(F0, specular_albedo, metallicity);

	float3 F = Fresnel_Schlick(fmaxf(dot(half_vec, -ray_dir), 0.0f), F0);
	float D = D_GGX(normal, half_vec, roughness); // normal distribution function for brdf
	float G = G_Smith(alpha, normal, -ray_dir, light_dir); // geometry function for brdf

    // combined specular brdf
	float3 BRDF_specular = (D * F * G) / (4.0f * fmaxf(dot(normal, -ray_dir), 0.0f) * fmaxf(dot(normal, light_dir), 0.0f));




	// solely diffuse lighting for now
    float3 light_dir_diffuse;
	cosine_sample_hemisphere(myrnd(seed), myrnd(seed), light_dir_diffuse);
	payload.direction = light_dir_diffuse;






	if (payload.depth <= 0) payload.done = true;

    payload.origin = hit_pos;
	payload.attenuation *= diffuse_albedo; // accumulates how much the material absorbs light on the given path

    payload.seed = seed; // update the seed to keeprandomness

    setPayloadCH(payload);
}
