// The SphereData structure holds the data for a single sphere in the scene.
struct TriangleData
{
	float4 v0, v1, v2, n0, n1, n2;
    // UV coordinates per vertex
	float2 uv0, uv1, uv2;
};

// The Params structure holds the scene parameters that are read by the ray generation program.
struct Params
{
    unsigned int           image_width;   // Width of the output image in pixels
    unsigned int           image_height;  // Height of the output image in pixels
    int                    origin_x;      // X-coordinate of the image origin (not used in this sample)
    int                    origin_y;      // Y-coordinate of the image origin (not used in this sample)
    int                    subframe_index;

    uchar4*                frame_buffer;
    float4*                accum_buffer;
    bool dof;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    TriangleData*          triangles;       // Pointer to the array of spheres in the scene
    unsigned int           num_triangles;   // Number of spheres in the scene

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

	bool specular_bounce;   // Whether the ray is a specular bounce or not
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
    float4* hdr_image_data; // Device pointer to the HDRi image data
    int     width;          // Image width
    int     height;         // Image height
};


// The HitGroupData structure would contain data about the materials or geometric properties if needed.
struct HitGroupData
{
    float4* albedo_texture_data;
    int     tex_width;
    int     tex_height;
    bool    has_texture;

    // Roughness map
    float4* roughness_texture_data;
    int     roughness_width;
    int     roughness_height;
    bool    has_roughness_map;

    // Normal map
    float4* normal_texture_data;
    int     normal_width;
    int     normal_height;
    bool    has_normal_map;

    // Metallic map
    float4* metallic_texture_data;
    int     metallic_width;
    int     metallic_height;
    bool    has_metallic_map;

    float2* texcoords;
    float4* vertices;
    float4* normals;

    float3 emission_color;
    float3 diffuse_color;
    float3 specular;
    float  roughness;
    bool   metallic;
    bool   transparent;
};