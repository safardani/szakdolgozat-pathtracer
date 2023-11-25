// The SphereData structure holds the data for a single sphere in the scene.
struct SphereData
{
    float3 center;
    float radius;

    float3 color;
};

// The Params structure holds the scene parameters that are read by the ray generation program.
struct Params
{
    uchar4*                image;         // Pointer to the output image buffer (RGBA uchar)
    unsigned int           image_width;   // Width of the output image in pixels
    unsigned int           image_height;  // Height of the output image in pixels
    int                    origin_x;      // X-coordinate of the image origin (not used in this sample)
    int                    origin_y;      // Y-coordinate of the image origin (not used in this sample)

    SphereData*            spheres;       // Pointer to the array of spheres in the scene
    unsigned int           num_spheres;   // Number of spheres in the scene

    OptixTraversableHandle handle;        // Handle to the top-level acceleration structure for raytracing
};

struct Payload {
    float3 result;
    float3 origin;
    float3 direction;
    float3 attenuation;
    unsigned int hit;
    int seed;
};

// The RayGenData structure is populated with data used by the ray generation program.
struct RayGenData
{
    float3 cam_eye;    // Position of the camera eye (origin of the rays)
    float3 camera_u;   // U component of the camera basis vector (controls horizontal field of view)
    float3 camera_v;   // V component of the camera basis vector (controls vertical field of view)
    float3 camera_w;   // W component of the camera basis vector (determines direction the camera is facing)
};


// The MissData structure contains data used by the miss program (the background color).
struct MissData
{
    float r, g, b;  // RGB color of the background
};


// The HitGroupData structure would contain data about the materials or geometric properties if needed.
struct HitGroupData
{
    float3 color; // Color of the sphere material.
};