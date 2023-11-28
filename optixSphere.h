// The SphereData structure holds the data for a single sphere in the scene.
struct SphereData
{
    float3 center;
    float radius;
    float3 color;
    float3 specular;   // Specular reflectance of the material.
    float  roughness;  // Roughness value of the material.
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
    float3 attenuation;     // Attenuation of the ray
    int seed;               // Seed for the random number generator

    float3 emitted; 	    // Emitted radiance
    float3 radiance; 	    // The amount of light that passes through a particular area and falls within a given solid angle in a specified direction
    float3 origin;          // Origin position of the ray
    float3 direction;       // Direction of the ray
    int done;
    int depth;              // Depth of the ray
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
    float3 emission_color;
    float3 diffuse_color;
    float3 specular;   // Specular reflectance of the material.
    float  roughness;  // Roughness value of the material.
};