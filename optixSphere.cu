#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>
#include <optix_device.h>

#include <sutil/vec_math.h>
#include <random.h>

#include <stdio.h>

#define RANDFLOAT3 make_float3(rnd(seed), rnd(seed), rnd(seed))

// Declare a constant Params structure that will be filled in by the host (CPU) before launch,
// and can be accessed by all the device (GPU) kernels.
extern "C" {
    __constant__ Params params;
}


// This utility function performs the actual ray tracing from the given origin in the given direction.
// If the ray intersects an object, the payload (prd) contains the shading information which is set in closest-hit or miss program.
static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,          // The traversable handle representing the scene to trace against
    float3                 ray_origin,      // The origin of the ray
    float3                 ray_direction,   // The direction of the ray
    float                  tmin,            // The minimum distance along the ray to check for intersections
    float                  tmax,            // The maximum distance along the ray to check for intersections
    Payload* prd                             // The payload registered with the ray
)
{
    unsigned int p0, p1, p2, o1, o2, o3, d1, d2, d3, a1, a2, a3, h, seed;

    p0 = __float_as_uint(prd->result.x);
    p1 = __float_as_uint(prd->result.y);
    p2 = __float_as_uint(prd->result.z);

    o1 = __float_as_uint(prd->origin.x);
    o2 = __float_as_uint(prd->origin.y);
    o3 = __float_as_uint(prd->origin.z);

    d1 = __float_as_uint(prd->direction.x);
    d2 = __float_as_uint(prd->direction.y);
    d3 = __float_as_uint(prd->direction.z);

    a1 = __float_as_uint(prd->attenuation.x);
    a2 = __float_as_uint(prd->attenuation.y);
    a3 = __float_as_uint(prd->attenuation.z);

    h = prd->hit;

    seed = prd->seed;

    // Perform the trace call which will call into the intersect, any-hit and closest-hit programs
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                   // rayTime: a value to simulate motion blur (not used here)
        OptixVisibilityMask(1), // Visibility mask to define which objects this ray should intersect
        OPTIX_RAY_FLAG_NONE,    // A set of flags that can be used to control ray behavior
        0,                      // SBT offset: Index into the Shader Binding Table
        1,                      // SBT stride: The step between records in the SBT used for consecutive rays
        0,                      // missSBTIndex: Index of the miss shader in the SBT
        p0, p1, p2,
        o1, o2, o3,
        d1, d2, d3,
        a1, a2, a3,
        h, seed           // payload
    );

    prd->result.x = __uint_as_float(p0);
    prd->result.y = __uint_as_float(p1);
    prd->result.z = __uint_as_float(p2);

    prd->origin.x = __uint_as_float(o1);
    prd->origin.y = __uint_as_float(o2);
    prd->origin.z = __uint_as_float(o3);

    prd->direction.x = __uint_as_float(d1);
    prd->direction.y = __uint_as_float(d2);
    prd->direction.z = __uint_as_float(d3);

    prd->attenuation.x = __uint_as_float(a1);
    prd->attenuation.y = __uint_as_float(a2);
    prd->attenuation.z = __uint_as_float(a3);

    prd->hit = h;

    prd->seed = seed;
}


// A helper function to set the payload for the current ray
static __forceinline__ __device__ void setPayload(Payload p)
{
    optixSetPayload_0(__float_as_uint(p.result.x));
    optixSetPayload_1(__float_as_uint(p.result.y));
    optixSetPayload_2(__float_as_uint(p.result.z));

    optixSetPayload_3(__float_as_uint(p.origin.x));
    optixSetPayload_4(__float_as_uint(p.origin.y));
    optixSetPayload_5(__float_as_uint(p.origin.z));

    optixSetPayload_6(__float_as_uint(p.direction.x));
    optixSetPayload_7(__float_as_uint(p.direction.y));
    optixSetPayload_8(__float_as_uint(p.direction.z));

    optixSetPayload_9(__float_as_uint(p.attenuation.x));
    optixSetPayload_10(__float_as_uint(p.attenuation.y));
    optixSetPayload_11(__float_as_uint(p.attenuation.z));

    optixSetPayload_12(p.hit);

    optixSetPayload_13(p.seed); 
}


// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ Payload getPayload()
{
    return Payload{
        make_float3(
            __uint_as_float(optixGetPayload_0()),
            __uint_as_float(optixGetPayload_1()),
            __uint_as_float(optixGetPayload_2())),
make_float3(
            __uint_as_float(optixGetPayload_3()),
            __uint_as_float(optixGetPayload_4()),
            __uint_as_float(optixGetPayload_5())),
        make_float3(
            __uint_as_float(optixGetPayload_6()),
            __uint_as_float(optixGetPayload_7()),
            __uint_as_float(optixGetPayload_8())),

        make_float3(
            __uint_as_float(optixGetPayload_9()),
            __uint_as_float(optixGetPayload_10()),
            __uint_as_float(optixGetPayload_11())),

        optixGetPayload_12(),
        (int)optixGetPayload_13()
    };
}


static __forceinline__ __device__ float3 random_in_unit_sphere(unsigned int seed) {
    float3 p;
    do {
        p = 2.0f * RANDFLOAT3 - make_float3(1, 1, 1);
    } while (p.x * p.x + p.y * p.y + p.z * p.z >= 1.0f);
    return p;
}

// Uncharted 2 filmic tonemap operator (by Hable, et al.)
static __forceinline__ __device__ float3 tonemap(float3 x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// The ray generation program. This is called once per pixel, and its job is to generate the primary rays.
extern "C" __global__ void __raygen__rg()
{
    // Compute the launch index, which corresponds to the pixel indices in the image
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Get a pointer to the ray generation data stored in the Shader Binding Table
    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    // Get the camera basis vectors and eye position from the ray generation data.
    const float3      U = rtData->camera_u;
    const float3      V = rtData->camera_v;
    const float3      W = rtData->camera_w;

    Payload payload = Payload{
        make_float3(1.0f, 1.0f, 1.0f),
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(1.0f, 1.0f, 1.0f), 1, 1
    };
    // Set an initial color payload for the ray which might be modified by the closest-hit or miss program
    float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    unsigned int seed = tea<4>(idx.y, idx.x);

    int sample_batch_count = 2000;
    for (size_t i = 0; i < sample_batch_count; i++)
    {
        seed = tea<4>(idx.y * 1600 + idx.x, i);
        payload.seed = seed;

        float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        // Calculate image plane location for the current ray.
        float2 d = 2.0f * make_float2((idx.x + subpixel_jitter.x) / (dim.x), (idx.y + subpixel_jitter.y) / (dim.y)) - 1.0f;

        // Calculate the ray origin and direction for the current pixel
        float3 origin = rtData->cam_eye;
        float3 direction = normalize(d.x * U + d.y * V + W);
        
        payload.attenuation = make_float3(1.0f, 1.0f, 1.0f);
        payload.direction = direction;

        // Trace the ray into the scene
        int max_depth = 20;
        for (int q = 0; q < max_depth; q++)
        {
            trace(
                params.handle,
                origin,
                direction,
                0.0005f,  // tmin
                1e16f,  // tmax
                &payload);

            if (payload.hit == 0) {
                payload_rgb += make_float3(payload.result.x, payload.result.y, payload.result.z);
                break;
            }

            origin = payload.origin;
            direction = payload.direction;
        }
    }

    // Apply filmic tonemapping to the HDR values
    payload_rgb = tonemap(payload_rgb / sample_batch_count);

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

static __forceinline__ __device__ float modified_sigmoid(float angle) {
    float exponent = (0.99f - angle) * 1100.f;
    float sigmoid = (19.8f / (1.f + expf(exponent))) + 0.2f;
    
    return sigmoid;
}

// The miss program. This is called for any ray that does not hit geometry.
extern "C" __global__ void __miss__ms()
{
    // Get a pointer to the miss program data stored in the Shader Binding Table.
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    // Read the existing payload. This payload was set by the raygen program.
    Payload payload = getPayload();

    float3 unit_direction = normalize(payload.direction);
    float angle_mult = dot(unit_direction, normalize(make_float3(0.9f, 1.0f, 3.0f))); 
    float3 c = modified_sigmoid(angle_mult) * make_float3(0.29f, 0.58f, 0.94f); 

    float3 sampled_color = payload.attenuation * c; 

    setPayload(Payload{
        sampled_color,
        payload.origin, payload.direction, make_float3(0.0f), 0, payload.seed
    });
}


// The closest hit program. This is called when a ray hits the closest geometry.
extern "C" __global__ void __closesthit__ch()
{
    // Optix gives us information about the intersected object and hit distance
    float  t_hit = optixGetRayTmax();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    // Other information such as primitive index, traversable handle and SBT GAS index
    const unsigned int           prim_idx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    // Define and retrieve the sphere's data (center and radius)
    float4 q;
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);
    
    Payload p = getPayload();

    // Retrieve the current HitGroupData from the SBT
    HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Compute the intersection point in world space
    float3 world_raypos = ray_orig + t_hit * ray_dir;
    // Transform the intersection point from world space to object space
    float3 obj_raypos = optixTransformPointFromWorldToObjectSpace(world_raypos);
    // Determine the object space normal, and then transform it back to world space
    float3 obj_normal = (obj_raypos - make_float3(q)) / q.w;
    // Normalize the normal vector after transforming it to world space
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));

    // Use the provided color from the hit group data
    const float3 sphere_color = hit_group_data->color;

    float3 target = world_raypos + world_normal + random_in_unit_sphere(p.seed);
    float3 new_dir = normalize(target - world_raypos);

    //float3 cur_attenuation = p.attenuation * (rt_data->color);
    float3 cur_attenuation = sphere_color * p.attenuation;

    // Set the payload to the resulting normal shaded color
    setPayload(Payload{
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(world_raypos.x, world_raypos.y, world_raypos.z),
        make_float3(new_dir.x, new_dir.y, new_dir.z), cur_attenuation, 1, p.seed
    });
}