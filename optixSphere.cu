#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
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
	unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18;

    u0 = __float_as_uint(prd->attenuation.x);
    u1 = __float_as_uint(prd->attenuation.y);
    u2 = __float_as_uint(prd->attenuation.z);
    u3 = prd->seed;
	u4 = __float_as_uint(prd->emitted.x);
	u5 = __float_as_uint(prd->emitted.y);
	u6 = __float_as_uint(prd->emitted.z);
	u7 = __float_as_uint(prd->radiance.x);
	u8 = __float_as_uint(prd->radiance.y);
	u9 = __float_as_uint(prd->radiance.z);
	u10 = __float_as_uint(prd->origin.x);
	u11 = __float_as_uint(prd->origin.y);
	u12 = __float_as_uint(prd->origin.z);
	u13 = __float_as_uint(prd->direction.x);
	u14 = __float_as_uint(prd->direction.y);
	u15 = __float_as_uint(prd->direction.z);
	u16 = prd->done;
    u17 = prd->depth;
	u18 = prd->specular_bounce;

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
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18);
    optixReorder(
        // Application specific coherence hints could be passed in here
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18);

    prd->attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd->seed = u3;
    
    prd->emitted = make_float3(__uint_as_float(u4), __uint_as_float(u5), __uint_as_float(u6));
    prd->radiance = make_float3(__uint_as_float(u7), __uint_as_float(u8), __uint_as_float(u9));
    prd->origin = make_float3(__uint_as_float(u10), __uint_as_float(u11), __uint_as_float(u12));
    prd->direction = make_float3(__uint_as_float(u13), __uint_as_float(u14), __uint_as_float(u15));
    prd->done = u16;
    prd->depth = u17;
	prd->specular_bounce = u18;
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
	p.specular_bounce = optixGetPayload_18();

    return p;
}

// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ Payload getPayloadMiss()
{
    Payload p = {};
	
    p.radiance.x = __uint_as_float(optixGetPayload_7());
    p.radiance.y = __uint_as_float(optixGetPayload_8());
    p.radiance.z = __uint_as_float(optixGetPayload_9());

	p.attenuation.x = __uint_as_float(optixGetPayload_0());
	p.attenuation.y = __uint_as_float(optixGetPayload_1());
	p.attenuation.z = __uint_as_float(optixGetPayload_2());

	p.depth = optixGetPayload_17();

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
	optixSetPayload_18(p.specular_bounce);
}


// A helper function to set the payload for the current ray
static __forceinline__ __device__ void setPayloadMiss(Payload p)
{
	optixSetPayload_0(__float_as_uint(p.attenuation.x));
	optixSetPayload_1(__float_as_uint(p.attenuation.y));
	optixSetPayload_2(__float_as_uint(p.attenuation.z));

    optixSetPayload_4(__float_as_uint(p.emitted.x));
    optixSetPayload_5(__float_as_uint(p.emitted.y));
    optixSetPayload_6(__float_as_uint(p.emitted.z));
    
    optixSetPayload_7(__float_as_uint(p.radiance.x));
    optixSetPayload_8(__float_as_uint(p.radiance.y));
    optixSetPayload_9(__float_as_uint(p.radiance.z));
    
    optixSetPayload_16(p.done);
	optixSetPayload_17(p.depth);
}

// A helper function to generate a random 3D vector that is distributed as a cosine-weighted hemisphere around the z-axis
static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.z = r * sinf(phi);

    // Project up to hemisphere.
    p.y = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.z * p.z));
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

    // TODO clean up
    unsigned int unique_key = idx.y * params.image_width + idx.x + subframe_index * params.image_width * params.image_height;
    unsigned int seed = unique_key; //hash(unique_key);

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

		float3 path_rgb = make_float3(0.0f);

        // Reset the payload data for this ray (every iteration of the loop)
        Payload payload;
        payload.attenuation = make_float3(1.0f);
        payload.radiance = make_float3(0.0f);
        payload.emitted = make_float3(0.0f);
		payload.specular_bounce = false;

        payload.seed = seed;
		payload.done = false;
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

            // get the colors from the payload
			// Accumulate the ray's contribution to the pixel color TODO rework the way this is accumulated

            //payload_rgb += payload.emitted;
            path_rgb = payload.radiance;


            // Russian roulette: try to keep path weights equal to one by randomly terminating paths.
			//*
			//const float p = dot(payload.attenuation, make_float3(0.30f, 0.59f, 0.11f)); // TODO which of these is correct
            const float p = fmaxf(payload.attenuation.x, fmaxf(payload.attenuation.y, payload.attenuation.z));
            const bool done = payload.done || myrnd(seed) > p;
            if (done && p > 0.0f) {
                path_rgb /= p; // TODO where does this go
                break;
            }
            //*/
			//if (payload.done) break;

            // Update ray origin and direction for the next path segment
            origin = payload.origin;
            direction = payload.direction;

            payload.depth--;
        }
        payload_rgb += path_rgb;
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
static __forceinline__ __device__ float D_GGX(float3 n, float3 h, float a)
{
    if (a <= 0.02) {
        if (dot(n, h) >= 0.999) {
            return 1;
        } else {
            return 0;
        }
    } else {
        float a2 = a * a;
        float NdotH = fmaxf(dot(n, h), 0.0000001f);
        float NdotH2 = NdotH * NdotH;

        float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
        denom = M_PIf * denom * denom;

	    if (denom < 0.0000001f)
		    return 1.0f;
        return a2 / denom;
    }

}

// Geometry Function (G) using Schlick-GGX
static __forceinline__ __device__ float GGX_PartialGeometryTerm(float3 v, float3 n, float3 h, float alpha)
{
    float VoH2 = clamp(dot(v, h), 0.0f, 1.0f);
    float chi = VoH2 / clamp(dot(v, n), 0.0f, 1.0f) > 0.0f ? 1.0f : 0.0f;
    VoH2 = VoH2 * VoH2;
    float tan2 = (1.0f - VoH2) / VoH2;

    return (chi * 2.0f) / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
}

// Geometry Function (G) using Schlick-GGX
static __forceinline__ __device__ float G_SchlickGGX(float alpha, float3 n, float3 x)
{
	float numerator = fabsf(dot(n, x));

	float k = alpha / 2.0f;
	float denominator = fabsf(dot(n, x)) * (1.0f - k) + k;
	denominator = fmaxf(denominator, 0.000001f);

	return numerator / denominator;
}

static __forceinline__ __device__ float G_Smith(float alpha, float3 N, float3 V, float3 L, float3 H)
{
    return G_SchlickGGX(alpha, N, V) * G_SchlickGGX(alpha, N, L);
}

// Fresnel-Schlick approximation
static __forceinline__ __device__ float3 Fresnel_Schlick(float cosTheta, float3 F0, float3 spec)
{
    cosTheta = clamp(cosTheta, 0.0f, 1.0f); // Ensure valid range
    return F0 + (make_float3(1) - F0) * powf(1.0f - cosTheta, 5.0f); // TODO spec or no spec
}

// Fresnel-Schlick approximation
static __forceinline__ __device__ float Fresnel_Schlick_float(float cosine, float refraction_index) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1.0f - cosine), 5.0f);
}

static __forceinline__ __device__ float3 GGX_importance_sample(float r1, float r2, float alpha) {
    float phi = 2.0f * M_PIf * r1;
    float cosTheta = sqrt((1.0f - r2) / (1.0f + (alpha * alpha - 1.0f) * r2));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    return normalize(make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi)));
}

// Custom bilinear texture sampling function
__device__ float4 sampleHDRI(float4* hdr_image_data, int width, int height, float u, float v)
{
    // Convert (u, v) to pixel coordinates
    float x = u * width - 0.5f;
    float y = v * height - 0.5f;

    int x0 = static_cast<int>(floorf(x)) % width;
    int y0 = static_cast<int>(floorf(y)) % height;
    int x1 = (x0 + 1) % width;
    int y1 = (y0 + 1) % height;

    float s = x - floorf(x);
    float t = y - floorf(y);

    // Fetch the four neighboring pixels
    float4 c00 = hdr_image_data[y0 * width + x0];
    float4 c10 = hdr_image_data[y0 * width + x1];
    float4 c01 = hdr_image_data[y1 * width + x0];
    float4 c11 = hdr_image_data[y1 * width + x1];

    // Bilinear interpolation
    float4 c0 = lerp(c00, c10, s);
    float4 c1 = lerp(c01, c11, s);
    float4 c = lerp(c0, c1, t);

    return c;
}

extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    // Retrieve the current MissData from the SBT
    const MissData* ms_data = reinterpret_cast<const MissData*>(optixGetSbtDataPointer());

    Payload prd = getPayloadMiss();

    float3 ray_dir = normalize(optixGetWorldRayDirection());

    // Map the ray direction to texture coordinates (u, v)
    float u = 0.5f + atan2f(ray_dir.z, ray_dir.x) / (2.0f * M_PIf);
    float v = 0.5f - asinf(ray_dir.y) / M_PIf;

    // Sample the HDRi image data
    float4 hdr_color = sampleHDRI(ms_data->hdr_image_data, ms_data->width, ms_data->height, u, v);

    // Use the sampled color as the radiance
    prd.radiance += prd.attenuation * make_float3(hdr_color.x, hdr_color.y, hdr_color.z);

    prd.emitted = make_float3(0.f);
    prd.done = true;

    setPayloadMiss(prd);
}

// The closest hit program. This is called when a ray hits the closest geometry.
extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    const unsigned int  prim_idx = optixGetPrimitiveIndex();
	const float3        ray_dir = optixGetWorldRayDirection(); // TODO what the hell is up with normalization // direction that the ray is heading in, from the origin
	//float3 ray_dir_original = optixGetWorldRayDirection();
    const float3        ray_orig = optixGetWorldRayOrigin();
	float               t_hit = optixGetRayTmax(); // distance to the hit point

	// get the geometry acceleration structure so we can get the sphere's properties
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    const int    vert_idx_offset = prim_idx * 3;

    const float3 v0 = make_float3(hit_group_data->vertices[vert_idx_offset + 0]);
    const float3 v1 = make_float3(hit_group_data->vertices[vert_idx_offset + 1]);
    const float3 v2 = make_float3(hit_group_data->vertices[vert_idx_offset + 2]);
    /*
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));

    const float3 normal = faceforward(N_0, -ray_dir, N_0);
    //*/
    Payload payload = getPayloadCH();

    float3 n0 = make_float3(hit_group_data->normals[vert_idx_offset + 0]);
    float3 n1 = make_float3(hit_group_data->normals[vert_idx_offset + 1]);
    float3 n2 = make_float3(hit_group_data->normals[vert_idx_offset + 2]);

    //*
    // Get barycentric coordinates
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float bary_beta = barycentrics.x;
    const float bary_gamma = barycentrics.y;
    const float bary_alpha = 1.0f - bary_beta - bary_gamma;
    
    // Interpolate normal
    float3 normal = bary_alpha * n0 + bary_beta * n1 + bary_gamma * n2;
    if (length(normal) > 0.01f) normal = normalize(normal);
    else {
        payload.done = true;
        setPayloadCH(payload);
        return;
    }
    //normal = faceforward(normal, -ray_dir, normal); 
    //*/

    const float3 hit_pos = ray_orig + t_hit * ray_dir;

	unsigned int seed = payload.seed;

    float3 specular_albedo = hit_group_data->specular;
	float3 diffuse_albedo = hit_group_data->diffuse_color;
	float3 emission_color = hit_group_data->emission_color;

	float roughness = hit_group_data->roughness; //roughness *= roughness;
	float metallicity = hit_group_data->metallic ? 1.0f : 0.0f;
	float transparency = hit_group_data->transparent ? 1.0f : 0.0f;
	float ior = 1.5f;

	if (payload.depth == 0)
        payload.emitted = emission_color; // TODO why are we doing this
    else
        payload.emitted = make_float3(0.0f);


    if (length(emission_color) > 0.0001f)
    {
		payload.radiance += payload.attenuation * emission_color;
		payload.done = true;
		setPayloadCH(payload);
		return;
	}

	random_in_unit_sphere(seed); // we need this FOR SOME REASON??? to keep the seed random and avoid artifacts. TODO

    if (roughness < 0.015f) roughness = 0.015f; // Prevent artifacts from division by very small numbers
	if (roughness > 0.999f) roughness = 0.999f; // Clamp roughness to 1.0

    if (payload.depth <= 0) payload.done = true;

	// GGX importance sampling code
    float r1 = myrnd(seed);
    float r2 = myrnd(seed);

    float alpha = roughness * roughness;
	float3 half_vec = GGX_importance_sample(r1, r2, alpha);

	// transform half vector from tangent space to world space
    Onb onb(normal);
    onb.inverse_transform(half_vec);
	normalize(half_vec);

    float3 light_dir = reflect(ray_dir, half_vec);
    float3 light_dir_diffuse;
	r1 = myrnd(seed);
	r2 = myrnd(seed);
    cosine_sample_hemisphere(r1, r2, light_dir_diffuse);
	onb.inverse_transform(light_dir_diffuse);

    float3 F0 = make_float3(fabs((1.0 - ior) / (1.0 + ior))); // 0.04 for dielectrics
    F0 = F0 * F0;
    F0 = lerp(F0, specular_albedo, metallicity);
    
    float3 F = Fresnel_Schlick(fmaxf(dot(normal, -ray_dir), 0.0f), F0, specular_albedo);
	float D = D_GGX(normal, half_vec, alpha); // normal distribution function for brdf
	float G = G_Smith(alpha, normal, -ray_dir, light_dir_diffuse, normalize(light_dir_diffuse - ray_dir)); // geometry function for brdf

    // combined specular brdf
	float3 brdf_specular = F * D * G / (4.0f * fabsf(dot(normal, -ray_dir)) * fabsf(dot(normal, light_dir_diffuse))); // TODO DIFFUSE OR NOT?
	//half_vec = normalize(light_dir_diffuse - ray_dir); // TODO is this correct? this is so the ggx sampling is ONLY affecting D, everything else is diffuse

    float NdotH = fmaxf(dot(normal, half_vec), 0.0f);
    float VdotH = fabsf(dot(-ray_dir, half_vec));
    VdotH = fmaxf(dot(-ray_dir, half_vec), 0.001f);
	float NdotV = fmaxf(dot(normal, - ray_dir), 0.0f);
    float IdotN = fabsf(dot(normal, normalize(light_dir_diffuse))); // TODO DIFFUSE OR NOT?
	float F_blend_factor = Fresnel_Schlick_float(NdotV, ior);

    // calculating the type of the next bounce
    float specular_probability = metallicity + (1.0f - metallicity) * F_blend_factor;
    //specular_probability = 1;
    if (myrnd(seed) < specular_probability)
    {
        payload.direction = normalize(light_dir);
        payload.specular_bounce = true;
    }
    else {

        payload.direction = normalize(light_dir_diffuse);
        payload.specular_bounce = false;
    }

	// probability distribution function for ggx importance sampling
	float spdf = D * NdotH / (4.0f * VdotH);
    float dpdf = 1.0f / M_PIf; // TODO IdotN gets pulled out to the attenuation part


    float pdf = specular_probability * spdf + (1.0f - specular_probability) * dpdf;
	float3 brdf = specular_probability * brdf_specular + (1.0f - specular_probability) * diffuse_albedo;

    // Determine if the material is transparent (glass)
    if (transparency > 0.5f)
    {
		// TODO need new way to check if the ray is inside the glass
        // Glass material handling
        float cos_theta_i = dot(normal, -ray_dir);
        float eta = ior; // Index of refraction of air
        
        float3 N = normal;
        
        // Check if the ray is inside the material
        if (cos_theta_i < 0.0f)
        {
            // Ray is inside the glass
            cos_theta_i = -cos_theta_i;
            N = -normal;
            // Swap eta_i and eta_t
            eta = 1.0f / eta;
        }
        
		// check for critical angle
        float sin_theta_t2 = eta * eta * (1.0f - cos_theta_i * cos_theta_i);
    
		// schlick's approximation
        float3 reflect_dir;
        float3 refract_dir;
        float reflectance = Fresnel_Schlick_float(cos_theta_i, ior);
		if (myrnd(seed) < reflectance)
        {
			float3 half_vec = GGX_importance_sample(r1, r2, alpha);
            Onb onb(normal);
            onb.inverse_transform(half_vec);
            normalize(half_vec);

            reflect_dir = reflect(ray_dir, half_vec);
			normalize(reflect_dir);
            payload.direction = reflect_dir;
			payload.specular_bounce = true;
		}
		else
		{
			// refract the ray
		    float alpha = roughness * roughness;
			refract(refract_dir, ray_dir, N, eta);
			normalize(refract_dir);
			payload.direction = refract_dir + 0.8f * alpha * random_in_unit_sphere(seed);
			payload.specular_bounce = false;
		}

        payload.origin = hit_pos;
        payload.seed = seed; // update the seed to keeprandomness
        setPayloadCH(payload);
		return;
    }

    //payload.radiance += payload.attenuation * (brdf * IdotN / pdf); // TODO needed when doing NEE
	if (length(brdf) >= 0.00001f)
        payload.attenuation *= (brdf * IdotN) / pdf;
    
    payload.origin = hit_pos;
    payload.seed = seed; // update the seed to keeprandomness
    
    setPayloadCH(payload);
}
