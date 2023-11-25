#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>
#include <optix_device.h>

#include <sutil/vec_math.h>

#include <stdio.h>

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
    float3* prd                             // The payload registered with the ray
)
{
    unsigned int p0, p1, p2;

    // Convert the payload from a float3 to three unsigned ints, each component separately
    p0 = __float_as_uint(prd->x);
    p1 = __float_as_uint(prd->y);
    p2 = __float_as_uint(prd->z);

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
        p0, p1, p2);            // Three parts of the payload passed as arguments

    // After the trace, convert the payload back to the float3 format
    prd->x = __uint_as_float(p0);
    prd->y = __uint_as_float(p1);
    prd->z = __uint_as_float(p2);
}


// A helper function to set the payload for the current ray
static __forceinline__ __device__ void setPayload(float3 p)
{
    // Convert float3 payload values to unsigned int and set them using optixSetPayload
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}


// A helper function to retrieve the payload for the current ray
static __forceinline__ __device__ float3 getPayload()
{
    // Get the payload values as unsigned ints and convert them back to float3 before returning
    return make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2())
    );
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

    // Calculate image plane location for the current ray.
    const float2      d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)
    ) - 1.0f;

    // Calculate the ray origin and direction for the current pixel
    const float3 origin = rtData->cam_eye;
    const float3 direction = normalize(d.x * U + d.y * V + W);

    // Set an initial color payload for the ray which might be modified by the closest-hit or miss program
    float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);

    // Trace the ray into the scene
    trace(params.handle,
        origin,
        direction,
        0.00f,  // tmin
        1e16f,  // tmax
        &payload_rgb);

    // Gamma correction factor
    float gamma = 2.2f;

    // Perform simple gamma correction before writing the color to the image buffer
    payload_rgb = make_float3(
        powf(payload_rgb.x, 1.0f / gamma),
        powf(payload_rgb.y, 1.0f / gamma),
        powf(payload_rgb.z, 1.0f / gamma));

    // Write the gamma-corrected pixel color to the image buffer
    params.image[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
}


// The miss program. This is called for any ray that does not hit geometry.
extern "C" __global__ void __miss__ms()
{
    // Get a pointer to the miss program data stored in the Shader Binding Table.
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    // Read the existing payload. This payload was set by the raygen program.
    float3    payload = getPayload();
    // Set the payload to the background color defined for this miss program.
    setPayload(make_float3(rt_data->r, rt_data->g, rt_data->b));
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

    // Compute the intersection point in world space
    float3 world_raypos = ray_orig + t_hit * ray_dir;
    // Transform the intersection point from world space to object space
    float3 obj_raypos = optixTransformPointFromWorldToObjectSpace(world_raypos);
    // Determine the object space normal, and then transform it back to world space
    float3 obj_normal = (obj_raypos - make_float3(q)) / q.w;
    // Normalize the normal vector after transforming it to world space
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));

    // Retrieve the current HitGroupData from the SBT
    HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Use the provided color from the hit group data
    const float3 sphere_color = hit_group_data->color;

    // ISSUE: I get the color of the first sphere for all spheres
    // DEBUG PRINT STATEMENTS: output the primitive index and the color it's supposed to be.
    // const unsigned int sbt_index = optixGetSbtGASIndex();
    // if (prim_idx != 0)
    //     printf("SBT Index: %u, Primitive Index: %u, Color: %f, %f, %f\n",
    //         sbt_index, prim_idx, hit_group_data->color.x, hit_group_data->color.y, hit_group_data->color.z);

    // Define the direction of the light (normalized directional vector)
    const float3 light_direction = normalize(make_float3(-1.0f, -1.0f, -1.0f)); // Directional light from top-left-front

    // Calculate the shading based on the half Lambert technique
    float light_intensity = 0.5 * dot(world_normal, -light_direction) + 0.5;

    // Set the payload to the resulting normal shaded color
    setPayload(sphere_color * light_intensity);
}