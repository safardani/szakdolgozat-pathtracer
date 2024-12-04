// Standard includes for OptiX and CUDA functionality
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

// Sample-specific configuration data
#include <sampleConfig.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// Utility headers to handle CUDA buffers, exceptions, and other general tasks
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

// Header for this specific program
#include "optixSphere.h"

// Standard C++ includes for formatting and IO
#include <iomanip>
#include <iostream>
#include <string>
#include <random>
#include <array>

// Camera and trackball handling for interaction
#include <sutil/Camera.h>
#include <sutil/Trackball.h>

// Window handling
#include <GLFW/glfw3.h>
#include <sutil/GLDisplay.h>
#include <fstream>

/**
    A Shader Binding Table (SBT) record is a data structure in OptiX to map the
    intersection of rays and geometry, to the appropriate shaders that should be executed.
    SBT records hold the function pointers and the data those shaders need:

    1. Header: Contains the shader identifiers that get executed when a ray hits the geometry.
    2. Data: User-defined data associated with that specific hit, like the surface's material properties.

    Template needed for creating SBT records for the different shader types (ray generation, miss, hit).

    @tparam T A struct representing the user-defined data payload for a specific shader type.
*/
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; // Aligned header for shader identifiers.
    T data; // User-defined data for the shader.
};

// Define specific SBT record types using the SbtRecord template with specific data structures
typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

bool resize_dirty = false;
bool minimized = false;

bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

struct Material // TODO move to .h
{
    float3 color;
    float3 specular;   // Specular reflectance of the material.

    float emission;
    float roughness;  // Roughness value of the material.
    bool metallic;     // Whether the material is metallic or not.
    bool transparent;  // Whether the material is transparent or not.

    bool   has_texture;       // True if this material uses a texture
    sutil::ImageBuffer albedo_image; // Texture object for albedo // TODO fix this?

    bool   has_roughness_map;
    sutil::ImageBuffer roughness_image;

    bool   has_normal_map;
    sutil::ImageBuffer normal_image;

    bool   has_metallic_map;
    sutil::ImageBuffer metallic_image;
};

// Camera state
bool dof = true;
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

// Configure the camera for the scene. Sets the eye position, look-at point, up direction, etc.
void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
    camera.setEye({ 0.0f, 2.0f, 6.0f });
    camera.setLookat({ 0.0f, 0.0f, 0.0f });
    camera.setUp(normalize(make_float3(0.0f, 1.0f, 0.0f)));
    camera.setFovY(50.0f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);

    camera.setAspectRatio((float)width / (float)height);
}


// Prints the command line usage instructions and exits the program
void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit(1);
}


// Callback function for OptiX to log messages. It formats and redirects messages to stderr.
static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

// Set up random number generation
std::random_device rd;
std::mt19937 rnd(rd());

// Random number generation between 0 and 1
float rnd_f() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rnd);
}

// Handles mouse button events
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        // Set the mouse button state and start tracking the position
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}

// Handles mouse movement events
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        // Orbit mode
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->image_width, params->image_height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        // Look-around mode
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->image_width, params->image_height);
        camera_changed = true;
    }
}

// Handles window resizing events
static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Ensure the window size is not too small
    sutil::ensureMinimumSize(res_x, res_y);
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
    params->image_width = res_x;
    params->image_height = res_y;
    camera_changed = true;
    resize_dirty = true;
}

// Handles minimizing and restoring the window
static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}

// Handles key press events
static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            // Quit
            glfwSetWindowShouldClose(window, true);
        }
        else if (key == GLFW_KEY_G)
        {
            // Toggle depth of field
            dof = !dof;
            camera_changed = true;
        }
    }
}

// Handles mouse scroll events
static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

// Checks if the camera has changed and updates the relevant parameters accordingly
void handleCameraUpdate(Params& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.image_width) / static_cast<float>(params.image_height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

// Checks if the window has been resized and updates the output buffer accordingly
void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.image_width, params.image_height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.image_width * params.image_height * sizeof(float4)
    ));
}

// Resets the accumulation buffer when needed, ensures parameter consistency
void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    if (params.dof != dof)
        params.dof = dof;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}

// TODO: comments for this function
void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    int framebuf_res_x = 0;
    int framebuf_res_y = 0;
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

void generateSphereMesh(const float3& center, float radius, int stacks, int slices, const Material& material, std::vector<TriangleData>& triangles)
{
    std::vector<float4> vertices;
    std::vector<float4> normals;

    // Generate vertices and normals
    for (int i = 0; i <= stacks; ++i)
    {
        float phi = M_PI * i / stacks;
        float y = radius * cosf(phi);
        float r = radius * sinf(phi);

        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * M_PI * j / slices;
            float x = r * cosf(theta);
            float z = r * sinf(theta);

            float4 vertex = make_float4(center.x + x, center.y + y, center.z + z, 1.0f);
            vertices.push_back(vertex);

            // Compute normal
            float4 normal = make_float4(normalize(make_float3(x, y, z)), 0);
            normals.push_back(normal);
        }
    }

    // Generate triangles and assign normals
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            // First triangle
            TriangleData tri1;
            tri1.v0 = vertices[first];
            tri1.v1 = vertices[second];
            tri1.v2 = vertices[first + 1];
            tri1.n0 = normals[first];
            tri1.n1 = normals[second];
            tri1.n2 = normals[first + 1];
            //tri1.material = material;
            triangles.push_back(tri1);

            // Second triangle
            TriangleData tri2;
            tri2.v0 = vertices[first + 1];
            tri2.v1 = vertices[second];
            tri2.v2 = vertices[second + 1];
            tri2.n0 = normals[first + 1];
            tri2.n1 = normals[second];
            tri2.n2 = normals[second + 1];
            //tri2.material = material;
            triangles.push_back(tri2);
        }
    }
}

void setUpImageTexture(bool &has_map, sutil::ImageBuffer &image, std::string filename, CUdeviceptr& gpu_buffer) {
    if (fileExists(filename))
    {
        has_map = true;
        image = sutil::loadImage(filename.c_str());
        // Convert to float4 if needed, just like albedo
		std::cout << "Loaded texture " << filename << std::endl;
    } else { has_map = false; std::cout << "No texture found for " << filename << std::endl; }
    if (has_map) {
        size_t num_pixels = image.width * image.height;
        float4* float_pixels_roughness = new float4[num_pixels];
        if (image.pixel_format == sutil::BufferImageFormat::UNSIGNED_BYTE4) {

            unsigned char* udata = static_cast<unsigned char*>(image.data);
            for (size_t i = 0; i < num_pixels; ++i) {
                float r = udata[i * 4 + 0] / 255.0f;
                float g = udata[i * 4 + 1] / 255.0f;
                float b = udata[i * 4 + 2] / 255.0f;
                float a = udata[i * 4 + 3] / 255.0f;
                float_pixels_roughness[i] = make_float4(r, g, b, a);
            }

            // Replace the original data pointer and format
            image.data = float_pixels_roughness;
            image.pixel_format = sutil::BufferImageFormat::FLOAT4;
        }
        size_t tex_size = num_pixels * sizeof(float4);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_buffer), tex_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(gpu_buffer),
            image.data,
            tex_size,
            cudaMemcpyHostToDevice
        ));
        // Store this pointer in a vector so we can assign it in the hit records later
    }
}


CUdeviceptr d_tex_data;
CUdeviceptr d_roughness_data;
CUdeviceptr d_normal_data;
CUdeviceptr d_metallic_data;

void createSceneGeometry(
    std::vector<TriangleData>& triangles,
    std::vector<Material>& sceneMaterials,
    std::vector<uint32_t>& g_mat_indices,
    bool loadFromFile = false,
    std::vector<std::string> filenames = {}
) {
    if (loadFromFile)
    {
        // Clear materials and g_mat_indices just in case
        sceneMaterials.clear();
        g_mat_indices.clear();

        // For each input file, we assign one material.
        // We'll pick a simple default material for all imported geometry from that file.
        // For example, a neutral gray material:
        // (If desired, you can vary color per file)
        float minHeight = 10.0f;

        // Index to the next material we add
        // sceneMaterials is empty initially, we start adding materials as we go.
        for (int i = 0; i < (int)filenames.size(); i++)
        {

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            std::string mtl_basepath = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixSphere\\";
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filenames[i].c_str(), mtl_basepath.c_str());

            if (!warn.empty()) {
                std::cerr << "TinyObjLoader Warning: " << warn << std::endl;
            }

            if (!err.empty()) {
                std::cerr << "TinyObjLoader Error: " << err << std::endl;
            }

            if (!ret) {
                throw std::runtime_error("Failed to load/parse .obj file.");
            }

            // Store current triangle count before we add new ones
            size_t startIndex = triangles.size();

            // Iterate over shapes and their faces
            for (const auto& shape : shapes)
            {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
                {
                    int fv = shape.mesh.num_face_vertices[f];
                    if (fv != 3) {
                        // Skip non-triangle faces
                        index_offset += fv;
                        continue;
                    }

                    float4 vertices[3];
                    float4 normals[3];
                    float2 uv[3]; // store UV per vertex of the triangle

                    // Iterate over vertices in the face
                    for (int v = 0; v < 3; v++)
                    {
                        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                        vertices[v] = make_float4(vx + (i - 0.f) * 4.f, vy, vz, 0.0f);

                        if (idx.normal_index >= 0) {
                            tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                            tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                            tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                            normals[v] = make_float4(normalize(make_float3(nx, ny, nz)), 0.0f);
                        }
                        else {
                            normals[v] = make_float4(0.0f, 1.0f, 0.0f, 0.0f); // fallback normal if none
                        }

                        
                        if (!attrib.texcoords.empty()) {
                            // idx is the tinyobj::index_t for the vertex
                            tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                            tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                            uv[v] = make_float2(tx, ty);
                        }
                        else {
                            // No texture coordinates, assign dummy UV
                            uv[v] = make_float2(0.0f, 0.0f);
                        }
                    }

					if (vertices[0].y < minHeight) minHeight = vertices[0].y;
					if (vertices[1].y < minHeight) minHeight = vertices[1].y;
					if (vertices[2].y < minHeight) minHeight = vertices[2].y;

                    TriangleData tri;
                    tri.v0 = vertices[0];
                    tri.v1 = vertices[1];
                    tri.v2 = vertices[2];
                    tri.n0 = normals[0];
                    tri.n1 = normals[1];
                    tri.n2 = normals[2];
					tri.uv0 = uv[0];
					tri.uv1 = uv[1];
					tri.uv2 = uv[2];

                    triangles.push_back(tri);

                    index_offset += fv;
                }
            }

            Material objMaterial;

            sutil::ImageBuffer albedo_image;
            bool have_texture = false;
            std::string tex_filename = filenames[i].substr(0, filenames[i].find_last_of('.')) + "_albedo.png";
            setUpImageTexture(
                have_texture, albedo_image,
                tex_filename, d_tex_data);

            sutil::ImageBuffer roughness_image;
            bool have_roughness_map = false;
            std::string roughness_filename = filenames[i].substr(0, filenames[i].find_last_of('.')) + "_roughness.png";
			setUpImageTexture(
                have_roughness_map, roughness_image,
                roughness_filename, d_roughness_data);

            sutil::ImageBuffer normal_image;
            bool have_normal_map = false;
            std::string normal_filename = filenames[i].substr(0, filenames[i].find_last_of('.')) + "_normal.png";
            setUpImageTexture(
                have_normal_map, normal_image,
                normal_filename, d_normal_data);

            sutil::ImageBuffer metallic_image;
            bool have_metallic_map = false;
            std::string metallic_filename = filenames[i].substr(0, filenames[i].find_last_of('.')) + "_metallic.png";
			setUpImageTexture(
                have_metallic_map, metallic_image,
				metallic_filename, d_metallic_data);

			// std::cout << "has albedo map: " << have_texture << std::endl;
			// std::cout << "has roughness map: " << objMaterial.has_roughness_map << std::endl;
			// std::cout << "has normal map: " << objMaterial.has_normal_map << std::endl;
			// std::cout << "has metallic map: " << objMaterial.has_metallic_map << std::endl;

            float3 color = make_float3(rnd_f(), rnd_f(), rnd_f()); // diff color per file
            float decider = rnd_f();
			if (have_texture || have_metallic_map || have_normal_map || have_roughness_map) {
                // If we have a texture, set a neutral fallback
                objMaterial.color = make_float3(1.0f, 1.0f, 1.0f);
                objMaterial.specular = make_float3(1.0f, 1.0f, 1.0f);
                objMaterial.emission = 0.0f;
                objMaterial.roughness = 1.0f; // something reasonable
                objMaterial.metallic = false;
                objMaterial.transparent = false;
				objMaterial.has_texture = have_texture;
				objMaterial.albedo_image = albedo_image;
				objMaterial.has_roughness_map = have_roughness_map;
				objMaterial.roughness_image = roughness_image;
				objMaterial.has_normal_map = have_normal_map;
				objMaterial.normal_image = normal_image;
				objMaterial.has_metallic_map = have_metallic_map;
				objMaterial.metallic_image = metallic_image;
            }
            else {
                // Existing random material generation code:
                float3 color = make_float3(rnd_f(), rnd_f(), rnd_f());
                float decider = rnd_f();
                objMaterial.color = color;
				objMaterial.specular = color;
				objMaterial.emission = decider < 0.1f ? 100.0f : 0.0f; // TODO this can fuck us
				objMaterial.roughness = rnd_f();
				objMaterial.metallic = decider > 0.5f && decider < 0.65f;
				objMaterial.transparent = false;
            }
            
            int currentMaterialIndex = (int)sceneMaterials.size();
            sceneMaterials.push_back(objMaterial);

            size_t endIndex = triangles.size();
            // All triangles from this file get the same material index
            for (size_t t = startIndex; t < endIndex; ++t)
            {
                g_mat_indices.push_back((uint32_t)currentMaterialIndex);
            }

            std::cout << "Loaded " << (endIndex - startIndex) << " triangles from " << filenames[i] << std::endl;
        }

        // After loading all files, we add the floor geometry
        Material floorMaterial = {
            make_float3(1.0f, 1.0f, 1.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            0.0f,
            1.0f,
            false,
            false
        };

        int floorMaterialIndex = (int)sceneMaterials.size();
        sceneMaterials.push_back(floorMaterial);

        float floor_y = minHeight - 10;
        float floor_size = 200.0f;

        float4 fv0 = make_float4(-floor_size, floor_y, -floor_size, 0.0f);
        float4 fv1 = make_float4(-floor_size, floor_y, floor_size, 0.0f);
        float4 fv2 = make_float4(floor_size, floor_y, -floor_size, 0.0f);
        float4 fv3 = make_float4(floor_size, floor_y, floor_size, 0.0f);

        float4 floorNormal = make_float4(0.0f, 1.0f, 0.0f, 0.0f);

        size_t floorStart = triangles.size();

        // First floor triangle
        {
            TriangleData floorTri1;
            floorTri1.v0 = fv0;
            floorTri1.v1 = fv1;
            floorTri1.v2 = fv2;
            floorTri1.n0 = floorNormal;
            floorTri1.n1 = floorNormal;
            floorTri1.n2 = floorNormal;
            triangles.push_back(floorTri1);
            g_mat_indices.push_back((uint32_t)floorMaterialIndex);
        }

        // Second floor triangle
        {
            TriangleData floorTri2;
            floorTri2.v0 = fv2;
            floorTri2.v1 = fv1;
            floorTri2.v2 = fv3;
            floorTri2.n0 = floorNormal;
            floorTri2.n1 = floorNormal;
            floorTri2.n2 = floorNormal;
            triangles.push_back(floorTri2);
            g_mat_indices.push_back((uint32_t)floorMaterialIndex);
        }

        std::cout << "Loaded models with " << triangles.size() << " triangles total." << std::endl;
    }
    else
    {
        // Clear materials and g_mat_indices just in case
        sceneMaterials.clear();
        g_mat_indices.clear();

        // Define your materials here so it's all self-contained:
        Material groundMaterial = {
            make_float3(0.5f, 0.5f, 0.5f), // color
            make_float3(1.0f, 1.0f, 1.0f), // specular
            0.0f, // emission
            0.8f, // roughness
            false, // metallic
            false  // transparent
        };

        Material redSphere = {
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(1.0f, 0.0f, 0.0f),
            0.0f, 0.0f, false, false
        };

        Material greenSphere = {
            make_float3(0.0f, 1.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            0.0f, 0.0f, false, false
        };

        Material blueSphere = {
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            0.0f, 0.0f, false, false
        };

        // Add them to sceneMaterials in a known order:
        // Index 0: ground
        // Index 1: red sphere
        // Index 2: green sphere
        // Index 3: blue sphere
        sceneMaterials.push_back(groundMaterial);
        sceneMaterials.push_back(redSphere);
        sceneMaterials.push_back(greenSphere);
        sceneMaterials.push_back(blueSphere);

        // Generate ground plane
        float ground_y = 0.0f;
        float plane_size = 10.0f;

        float4 v0 = make_float4(-plane_size, ground_y, -plane_size, 1.0f);
        float4 v1 = make_float4(-plane_size, ground_y, plane_size, 1.0f);
        float4 v2 = make_float4(plane_size, ground_y, -plane_size, 1.0f);
        float4 v3 = make_float4(plane_size, ground_y, plane_size, 1.0f);

        float4 groundNormal = make_float4(0.0f, 1.0f, 0.0f, 0.0f);

        TriangleData tri1;
        tri1.v0 = v0; tri1.v1 = v1; tri1.v2 = v2;
        tri1.n0 = groundNormal; tri1.n1 = groundNormal; tri1.n2 = groundNormal;
        triangles.push_back(tri1);
        g_mat_indices.push_back(0); // ground material index is 0

        TriangleData tri2;
        tri2.v0 = v2; tri2.v1 = v1; tri2.v2 = v3;
        tri2.n0 = groundNormal; tri2.n1 = groundNormal; tri2.n2 = groundNormal;
        triangles.push_back(tri2);
        g_mat_indices.push_back(0); // ground material index is 0

        // Generate spheres
        int numSpheres = 3;
        float3 sphereCenters[] = {
            make_float3(-3.0f, 1.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            make_float3(3.0f, 1.0f, 0.0f)
        };

        for (int i = 0; i < numSpheres; ++i)
        {
            float3 center = sphereCenters[i];
            float radius = 1.0f;
            int stacks = 16;
            int slices = stacks * 2;

            // Store current triangle count before adding this sphere
            size_t startIndex = triangles.size();

            generateSphereMesh(center, radius, stacks, slices, /*unused material*/groundMaterial, triangles);

            // Assign sphere material indices
            // Each sphere gets a different material index: red=1, green=2, blue=3
            uint32_t sphereMatIndex = 1 + i;
            size_t endIndex = triangles.size();
            for (size_t t = startIndex; t < endIndex; ++t)
            {
                g_mat_indices.push_back(sphereMatIndex);
            }
        }

        // After this function completes, `sceneMaterials` has all materials,
        // `triangles` has all geometry, and `g_mat_indices` has a material index
        // for every triangle in the exact order they were pushed.
        std::cout << "Generated " << triangles.size() << " triangles (spheres and ground plane)." << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // Variables to hold the output file parameters and default image dimensions
    std::string outfile;
    // Debug mode: smaller image size for faster rendering
#if defined( NDEBUG )
    int         width = 1600;
    int         height = 1200;
#else
    int         width = 600;
    int         height = 400;
#endif

    // Parse command-line arguments to adjust the output file and image dimensions if needed
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                outfile = argv[++i];
            }
            else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 6) == "--dim=") {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        }
        else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA by freeing a dummy allocation - ensures that runtime is initialized
            CUDA_CHECK(cudaFree(0));

            CUcontext cuCtx = 0;  // zero means take the current context

            // Initialize the OptiX API
            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb; // Specify the logging callback function
            options.logCallbackLevel = 4; // Set the verbosity level for logging
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context)); // Create the OptiX device context
        }




        bool consistent_generation = false; // Flag to check if the spheres are consistently generated (debug and comparison purposes)
        //
        // Building an acceleration structure to represent the geometry in the scene (Acceleration Handling)
        //
        std::vector<TriangleData> triangles; // Vector to hold the spheres in the scene
        OptixTraversableHandle gas_handle; // Handle to the GAS (Geometry Acceleration Structure) that will be built
        CUdeviceptr            d_gas_output_buffer; // Device pointer for the buffer that will store the GAS
        CUdeviceptr d_vertex_buffer;
        CUdeviceptr d_normal_buffer;
		CUdeviceptr d_texcoord_buffer;
        CUdeviceptr  d_mat_indices = 0;

        bool loadFromFile = true; // Flag to load geometry from an OBJ file
        std::string projectPath = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixSphere\\";
        std::vector<std::string> objFilename = {
            projectPath + "test.obj"
        };
        std::string hdr_filename = projectPath + "env4.exr"; // Replace with your HDRi image path.
        sutil::ImageBuffer hdr_image = sutil::loadImage(hdr_filename.c_str());

        // Create scene geometry: spheres and ground plane
        std::vector<Material> sceneMaterials;
        std::vector<uint32_t> g_mat_indices;
        createSceneGeometry(triangles, sceneMaterials, g_mat_indices, /* loadFromFile */ loadFromFile, objFilename);
#define NUM_TRIANGLES triangles.size()

        // create a g_vertices array of all the vertices
        std::vector<float4> g_vertices;
        std::vector<float4> g_normals;
		std::vector<float2> g_texcoords;
        for (int i = 0; i < NUM_TRIANGLES; ++i) {
            g_vertices.push_back(triangles[i].v0);
            g_vertices.push_back(triangles[i].v1);
            g_vertices.push_back(triangles[i].v2);
            g_normals.push_back(triangles[i].n0);
            g_normals.push_back(triangles[i].n1);
            g_normals.push_back(triangles[i].n2);
			g_texcoords.push_back(triangles[i].uv0);
			g_texcoords.push_back(triangles[i].uv1);
			g_texcoords.push_back(triangles[i].uv2);
        }

        {
            // Define the build options for the acceleration structure
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            // Create an array of indices for the materials of the spheres
            //for (int i = 0; i < NUM_TRIANGLES; ++i) {
            //    g_mat_indices.push_back(i);
            //}

            // Allocate device memory for the array of material indices and copy the data from host to device
            const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_mat_indices),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
            ));

            // Allocate device memory for the array of spheres' center vertices and copy the data from host to device
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer), g_vertices.size() * sizeof(float4)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buffer), g_vertices.data(),
                g_vertices.size() * sizeof(float4), cudaMemcpyHostToDevice));
            
            // Allocate device memory for the array of spheres' center vertices and copy the data from host to device
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_normal_buffer), g_normals.size() * sizeof(float4)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_normal_buffer), g_normals.data(),
                g_normals.size() * sizeof(float4), cudaMemcpyHostToDevice));

            // Allocate device memory for the array of spheres' center vertices and copy the data from host to device
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texcoord_buffer), g_texcoords.size() * sizeof(float2)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_texcoord_buffer), g_texcoords.data(),
                g_texcoords.size() * sizeof(float2), cudaMemcpyHostToDevice));

            // Configure the build input to describe the spheres with the provided vertex and radius buffers
            std::vector<uint32_t> triangle_input_flags;

            for (int i = 0; i < NUM_TRIANGLES; ++i) {
                triangle_input_flags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
            }

            OptixBuildInput triangle_input = {};
            triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(float4);
            triangle_input.triangleArray.numVertices = static_cast<uint32_t>(g_vertices.size());
            triangle_input.triangleArray.vertexBuffers = &d_vertex_buffer;
            triangle_input.triangleArray.flags = triangle_input_flags.data();
            triangle_input.triangleArray.numSbtRecords = sceneMaterials.size();
            triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices; // Buffer with the material indices
            triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t); // Size of the material indices
            triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t); // Stride of the material indices

            // Compute the memory required for the GAS
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

            // Allocate temporary buffer for GAS build
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));


            // Allocate a buffer to store the output GAS and compacted size information
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                compactedSizeOffset + 8));

            // Define an OptixAccelEmitDesc to request the size of the compacted GAS after build
            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

            // TODO this is the issue for memory
            // Build the actual GAS and retrieve the compacted size
            OPTIX_CHECK(optixAccelBuild(context,
                0,  // CUDA stream
                &accel_options, &triangle_input,
                1,  // Number of build inputs
                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                &emitProperty,  // Emitted property list
                1               // Number of emitted properties
            ));


            // Assign the buffer for GAS output
            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

            // Free temporary buffers
            CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
            //CUDA_CHECK(cudaFree((void*)d_radius_buffer));

            // If compacted size is smaller, create a buffer of that size and do the compaction
            size_t compacted_gas_size;
            CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

            // If the GAS is not yet compacted, do so now
            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));
                OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));
                CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
            }
            else {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // Create OptiX module
        // 
        // A module is a compilation unit of shaders. It's made from a CUDA source file that contains
        // one or more shaders (e.g., ray generation, miss, closest-hit, any-hit, and intersection shaders).
        // The compile options control the inlining and optimization levels during JIT compilation,
        // while pipeline compile options control the characteristics of the whole pipeline, like the
        // number of payload and attribute values, motion blur or primitive types etc.
        //
        OptixModule module = nullptr;
        OptixModule triangle_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
            // Development builds: Favor debugging (less inlining, full debug information)
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            // Specify pipeline compile options that are constant for all the modules in the pipeline.
            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues = 19;
            pipeline_compile_options.numAttributeValues = 1;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            // Load CUDA source from file and create the module containing the compiled CUDA functions (shaders)
            size_t      inputSize = 0;
            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixSphere.cu", inputSize);
            OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, input,
                inputSize, LOG, &LOG_SIZE, &module));

            // Built-in intersection module creation for spheres:
            // Apart from user-defined intersection programs, OptiX also provides built-in intersection programs for certain primitives like spheres and triangles.
            OptixBuiltinISOptions builtin_is_options = {};
            builtin_is_options.usesMotionBlur = false;
            builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
            OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options,
                &builtin_is_options, &triangle_module));
        }

        //
        // Program group creation
        //
        // The program group is a collection of shaders that are bound together during pipeline execution.
        // Different raytracing stages (ray generation, intersection, any-hit, closest-hit, miss, etc.) are grouped.
        // These groups serve as the execution domains for the previously compiled shaders.
        // Each group is associated with specific shader types and gets the relevant module functions' names. 
        // 
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

            // Ray generation program group consists of shaders that generate rays and are executed at the start of the raytracing process
            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &raygen_prog_group_desc,
                1,   // Number of program groups being created
                &program_group_options,
                LOG, &LOG_SIZE,
                &raygen_prog_group
            ));

            // Miss program group consists of shaders executed when a ray fails to intersect any geometry in the scene
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &miss_prog_group_desc,
                1,   // Number of program groups being created
                &program_group_options,
                LOG, &LOG_SIZE,
                &miss_prog_group
            ));

            // Hit group program group encapsulates the closest-hit and optional any-hit shaders.
            // These are responsible for determining ray behavior upon interacting with geometry, such as computing color or spawning new rays.
            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
            hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS = triangle_module; // Use the built-in sphere module for intersection 
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hitgroup_prog_group_desc,
                1,   // Number of program groups being created
                &program_group_options,
                LOG, &LOG_SIZE,
                &hitgroup_prog_group
            ));
        }

        //
        // Link and create the OptiX pipeline that contains the configuration of the ray tracing stages
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth = 1; // Maximum trace recursion depth. Set to 1 as we are not implementing recursive ray tracing in this sample
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group }; // Array with all created program groups

            // Define linking options for the pipeline
            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth; // Set the maximum trace depth
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context,
                &pipeline_compile_options, // Compile options set earlier
                &pipeline_link_options, // Linking options
                program_groups, // Our program groups for raygen, miss and hit
                sizeof(program_groups) / sizeof(program_groups[0]), // Number of program groups
                LOG, &LOG_SIZE,
                &pipeline // The created pipeline
            ));

            // Calculate the amount of stack size needed for the pipeline execution
            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
            {
                // Accumulate stack size requirement for each program group
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
            }

            // Calculate final stack sizes
            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                0,  // No direct callables in recursion
                0,  // No continuation callables in recursion
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state, &continuation_stack_size));

            // Set the computed stack sizes onto the pipeline
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state, continuation_stack_size,
                1  // Maximum depth of traversable graph for callables
            ));
        }

        //
        // Set up shader binding table (SBT)
        // 
        // A Shader Binding Table (SBT) record is a data structure in OptiX to map the
        // intersection of rays and geometry, to the appropriate shaders that should be executed.
        // SBT records hold the function pointers and the data those shaders need :
        //
        OptixShaderBindingTable sbt = {};
        float4* d_hdr_image_data = nullptr;
        {
            // The ray generation record encapsulates the data structure which will
            // be read by the ray generation program upon execution. In this OptiX framework,
            // the ray generation program is responsible for initiating rays into the scene.
            // Each pixel of the to-be-rendered image corresponds to one invocation of the
            // ray generation shader, which will then emit a ray into the scene.
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
            sutil::Camera cam;
            configureCamera(cam, width, height);

            // Initialize the ray generation record with the camera data. This structure will be
            // used in the ray generation program to compute each ray's parameters based on pixel coordinates.
            RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            rg_sbt.data.cam_eye = cam.eye();
            cam.UVWFrame(rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w);

            // Pack the raygen_prog_group's shader identifier into the header of the ray generation record.
            // This header is used by OptiX to identify which shader program to execute for the rays
            // generated by this specific record (i.e., which ray generation shader function to invoke).
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            // Allocate device memory for the HDR image data
            size_t hdr_image_size = hdr_image.width * hdr_image.height * sizeof(float4);
            CUDA_CHECK(cudaMalloc(&d_hdr_image_data, hdr_image_size));
            CUDA_CHECK(cudaMemcpy(
                d_hdr_image_data,
                hdr_image.data,
                hdr_image_size,
                cudaMemcpyHostToDevice
            ));

            // Allocate device memory for the miss record
            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));

            // Initialize the miss SBT record
            MissSbtRecord ms_sbt;
            ms_sbt.data.hdr_image_data = d_hdr_image_data; // Set the device pointer
            ms_sbt.data.width = hdr_image.width;
            ms_sbt.data.height = hdr_image.height;

            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(miss_record),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            // Hit group record setup - contains the closest hit shader
            CUdeviceptr hitgroup_records;
            size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_records), hitgroup_record_size * sceneMaterials.size()));

            // Populate the SBT records with each sphere's color data
            HitGroupSbtRecord* hg_sbts = new HitGroupSbtRecord[sceneMaterials.size()];

            // Populate the hit group records with their associated data
            for (size_t i = 0; i < sceneMaterials.size(); ++i)
            {
                // Pack the hitgroup_prog_group's shader identifier into each hit group record's header.
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbts[i]));

                hg_sbts[i].data.vertices = reinterpret_cast<float4*>(d_vertex_buffer);
                hg_sbts[i].data.normals = reinterpret_cast<float4*>(d_normal_buffer);
				hg_sbts[i].data.texcoords = !g_texcoords.empty() ? reinterpret_cast<float2*>(d_texcoord_buffer) : nullptr;

                // Fill each of your SBT records with the appropriate color
                const Material& mat = sceneMaterials[i];
                hg_sbts[i].data.emission_color = mat.color * mat.emission;
                hg_sbts[i].data.diffuse_color = mat.color;
                hg_sbts[i].data.specular = mat.specular;
                hg_sbts[i].data.roughness = mat.roughness;
                hg_sbts[i].data.metallic = mat.metallic;
                hg_sbts[i].data.transparent = mat.transparent;

                if (mat.has_texture) {
                    hg_sbts[i].data.albedo_texture_data = reinterpret_cast<float4*>(d_tex_data);
                    hg_sbts[i].data.tex_width = (int)mat.albedo_image.width;
                    hg_sbts[i].data.tex_height = (int)mat.albedo_image.height;
                    hg_sbts[i].data.has_texture = true;
                }
                else {
                    hg_sbts[i].data.albedo_texture_data = nullptr;
					hg_sbts[i].data.tex_width = 0; hg_sbts[i].data.tex_height = 0;
					hg_sbts[i].data.has_texture = false;
                }
                if (mat.has_roughness_map) {
					hg_sbts[i].data.roughness_texture_data = reinterpret_cast<float4*>(d_roughness_data);
					hg_sbts[i].data.roughness_width = (int)mat.roughness_image.width;
					hg_sbts[i].data.roughness_height = (int)mat.roughness_image.height;
					hg_sbts[i].data.has_roughness_map = true;
				}
				else {
					hg_sbts[i].data.roughness_texture_data = nullptr;
					hg_sbts[i].data.roughness_width = 0; hg_sbts[i].data.roughness_height = 0;
					hg_sbts[i].data.has_roughness_map = false;
				}
                if (mat.has_normal_map) {
                    hg_sbts[i].data.normal_texture_data = reinterpret_cast<float4*>(d_normal_data);
                    hg_sbts[i].data.normal_width = (int)mat.normal_image.width;
                    hg_sbts[i].data.normal_height = (int)mat.normal_image.height;
                    hg_sbts[i].data.has_normal_map = true;
                }
				else {
					hg_sbts[i].data.normal_texture_data = nullptr;
					hg_sbts[i].data.normal_width = 0; hg_sbts[i].data.normal_height = 0;
					hg_sbts[i].data.has_normal_map = false;
				}
                if (mat.has_metallic_map) {
					hg_sbts[i].data.metallic_texture_data = reinterpret_cast<float4*>(d_metallic_data);
					hg_sbts[i].data.metallic_width = (int)mat.metallic_image.width;
					hg_sbts[i].data.metallic_height = (int)mat.metallic_image.height;
					hg_sbts[i].data.has_metallic_map = true;
				}
                else {
					hg_sbts[i].data.metallic_texture_data = nullptr;
					hg_sbts[i].data.metallic_width = 0; hg_sbts[i].data.metallic_height = 0;
					hg_sbts[i].data.has_metallic_map = false;
				}
            }

            // Copy the hit group SBT records to the device
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hitgroup_records),
                hg_sbts,
                hitgroup_record_size * sceneMaterials.size(),
                cudaMemcpyHostToDevice
            ));

            delete[] hg_sbts; // Clean up host memory

            // Fill Shader Binding Table structure
            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = hitgroup_records;
            sbt.hitgroupRecordStrideInBytes = hitgroup_record_size;
            sbt.hitgroupRecordCount = sceneMaterials.size();
        }

        // Create an output buffer for rendering the final image
        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

        std::string outfile;
        sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        int samples_per_launch = 0;

        // Set up launch parameters
        Params params;
        params.image_width = width;
        params.image_height = height;
        params.accum_buffer = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&params.accum_buffer),
            params.image_width * params.image_height * sizeof(float4)
        ));

        params.origin_x = width / 2;
        params.origin_y = height / 2;
        params.handle = gas_handle;
        params.subframe_index = 0u;
        params.frame_buffer = nullptr;
        params.dof = true;

        // Allocate device memory for the Params structure and copy from host to device
        CUdeviceptr d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice
        ));

        // Go through the command line arguments and adjust parameters if needed
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h")
            {
                printUsageAndExit(argv[0]);
            }
            else if (arg == "--no-gl-interop")
            {
                output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
            }
            else if (arg == "--file" || arg == "-f")
            {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                outfile = argv[++i];
            }
            else if (arg.substr(0, 6) == "--dim=")
            {
                const std::string dims_arg = arg.substr(6);
                int w, h;
                sutil::parseDimensions(dims_arg.c_str(), w, h);
                params.image_width = w;
                params.image_height = h;
            }
            else if (arg == "--launch-samples" || arg == "-s")
            {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                samples_per_launch = atoi(argv[++i]);
            }
            else
            {
                std::cerr << "Unknown option '" << argv[i] << "'\n";
                printUsageAndExit(argv[0]);
            }
        }



        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixPathTracer", params.image_width, params.image_height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            //
            // Render loop
            //
            {
                // TODO: annotate the render loop better
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    params.image_width,
                    params.image_height
                );

                output_buffer.setStream(stream);
                sutil::GLDisplay gl_display;

                // Rendering statistics
                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    // Launch
                    uchar4* result_buffer_data = output_buffer.map();
                    params.frame_buffer = result_buffer_data;
                    CUDA_CHECK(cudaMemcpyAsync(
                        reinterpret_cast<void*>(d_param),
                        &params, sizeof(Params),
                        cudaMemcpyHostToDevice, stream
                    ));

                    OPTIX_CHECK(optixLaunch(
                        pipeline,
                        stream,
                        (d_param),
                        sizeof(Params),
                        &sbt,
                        params.image_width,   // launch width
                        params.image_height,  // launch height
                        1                     // launch depth
                    ));
                    output_buffer.unmap();
                    CUDA_SYNC_CHECK();
                    // iterate through the pixels

                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++params.subframe_index;

                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        { // TODO: this branch is probably not needed, get rid of all the command line argument parsing
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    params.image_width,
                    params.image_height
                );

                handleCameraUpdate(params);
                handleResize(output_buffer, params);

                // Launch
                uchar4* result_buffer_data = output_buffer.map();
                params.frame_buffer = result_buffer_data;
                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void*>(d_param),
                    &params, sizeof(Params),
                    cudaMemcpyHostToDevice, stream
                ));

                OPTIX_CHECK(optixLaunch(
                    pipeline,
                    stream,
                    (d_param),
                    sizeof(Params),
                    &sbt,
                    params.image_width,   // launch width
                    params.image_height,  // launch height
                    1                     // launch depth
                ));
                output_buffer.unmap();
                CUDA_SYNC_CHECK();

                sutil::ImageBuffer buffer;
                buffer.data = output_buffer.getHostPointer();
                buffer.width = output_buffer.width();
                buffer.height = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage(outfile.c_str(), buffer, false);
            }

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        //
        // Cleanup
        //
        {
            // Free resources allocated for the SBT
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
            CUDA_CHECK(cudaFree((void*)d_vertex_buffer)); // TODO CHANGE TO VERTEX
            CUDA_CHECK(cudaFree((void*)d_normal_buffer)); // TODO CHANGE TO NORMAL
			CUDA_CHECK(cudaFree((void*)d_texcoord_buffer)); // TODO CHANGE TO TEXCOORD
			CUDA_CHECK(cudaFree((void*)d_tex_data));
			CUDA_CHECK(cudaFree((void*)d_roughness_data));
			CUDA_CHECK(cudaFree((void*)d_metallic_data));
			CUDA_CHECK(cudaFree((void*)d_normal_data));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));
            CUDA_CHECK(cudaFree(d_hdr_image_data));

            // Free the GAS output buffer
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

            // Destroy the OptiX objects created
            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));
            OPTIX_CHECK(optixModuleDestroy(triangle_module));

            // Finally destroy the OptiX context itself
            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception& e)
    {
        // Catch any exceptions, output the error message and exit with an error code
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0; // Normal program termination with code 0
}