// Standard includes for OptiX and CUDA functionality
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

// Sample-specific configuration data
#include <sampleConfig.h>

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

// Camera and trackball handling for interaction
#include <sutil/Camera.h>
#include <sutil/Trackball.h>

// Window handling
#include <GLFW/glfw3.h>
#include <sutil/GLDisplay.h>



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
    camera.setEye({ 2.0f, 3.0f, 11.0f });
    camera.setLookat({ 0.0f, 0.0f, 4.0f });
    camera.setUp(normalize(make_float3(0.0f, 1.0f, 0.0f )));
    camera.setFovY(90.0f);
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
        } else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                outfile = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg.substr(0, 6) == "--dim=") {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        } else {
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

		bool consistent_generation = true; // Flag to check if the spheres are consistently generated (debug and comparison purposes)
        //
        // Building an acceleration structure to represent the geometry in the scene (Acceleration Handling)
        //
        std::vector<SphereData> spheres; // Vector to hold the spheres in the scene
        OptixTraversableHandle gas_handle; // Handle to the GAS (Geometry Acceleration Structure) that will be built
        CUdeviceptr            d_gas_output_buffer; // Device pointer for the buffer that will store the GAS
        {
            // Define the build options for the acceleration structure
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            // Set the number of spheres to generate
            #define NUM_SPHERES 16

            // Generate random spheres, each with a random position, but on the same horizontal plane
            for (int i = 0; i < NUM_SPHERES; ++i) {
                bool overlap = false;
				SphereData sphere;
                if (i == 0) {
                    // Create a large sphere to act as the ground plane
					sphere.center = make_float3(0.0f, -1000.5f, 0.0f);
                    sphere.radius = 1000.f;
                    sphere.color = make_float3(0.2f);
                    sphere.specular = sphere.color;
                    sphere.roughness = 0.8f;
                    sphere.metallic = false;
                    sphere.transparent = false;
                    sphere.emission = 0.0f;
				} else if (!consistent_generation) {
                    float type_gen = rnd_f();
                    sphere.emission = 0.0f;

                    if (type_gen < 0.2f) {
						// Create a transparent sphere
						sphere.metallic = false;
						sphere.transparent = true;
                        if (type_gen < 0.1f) {
							sphere.roughness = 0.025f;
						} else {
							sphere.roughness = 0.35f;
						}
					} else if (type_gen < 0.6f) {
						// Create a metallic sphere
						sphere.metallic = true;
						sphere.transparent = false;
                        if (type_gen < 0.4f) {
                            sphere.roughness = type_gen - 0.2f;
                        } else {
                            sphere.roughness = 0.4f + 2.0f * (type_gen - 0.4f);
                        }
					} else {
						// Create a non-metallic, non-transparent sphere
						sphere.metallic = false;
						sphere.transparent = false;
                        if (type_gen < 0.8f) {
                            sphere.roughness = (type_gen - 0.2f) * 0.5f;
                        } else if (type_gen < 0.9f) {
                            sphere.roughness = 1.0f;
                            sphere.emission = 100.0f;
                        } else {
                            sphere.roughness = 0.4f + 2.0f * (type_gen - 0.4f);
                        }
					}

                    // Create random spheres
                    sphere.center = make_float3(15.0f * (rnd_f() - .5f), 0.0f, 15.0f * (rnd_f() - .5f));
				    sphere.radius = 0.5f;
                    sphere.color = make_float3(rnd_f(), rnd_f(), rnd_f());
                    sphere.specular = sphere.color;

                    // Check for overlapping spheres
                    for (int j = 0; j < spheres.size(); ++j) {
						float3 diff = spheres[j].center - sphere.center;
						float dist = sqrtf(dot(diff, diff));

						if (dist < spheres[j].radius + sphere.radius) {
							--i;
                            overlap = true;
						}
					}
                } else {
					// Have a row of metallic, non-metallic, and transparent spheres
					// Each row should have 5 spheres lined up with increasing roughness (0, 0.25, 0.5, 0.75, 1.0)

					int row = (i - 1) / 3; // dictates the roughness of the sphere
					int col = (i - 1) % 3; // dictates if the sphere is metallic, transparent, or neither
                    
					sphere.center = make_float3(2.0f * (col - 2), 0.0f, 2.0f * row);
					sphere.radius = 0.5f;
					sphere.color = make_float3(0.35f, 0.55f, 0.4f);
					sphere.specular = sphere.color;
					sphere.emission = 0.0f;
                    
					if (col == 0) {
						sphere.metallic = false;
						sphere.transparent = false;
                        sphere.roughness = 0.25f * row;
					}
					else if (col == 1) {
						sphere.metallic = true;
						sphere.transparent = false;
						sphere.roughness = 0.25f * row;
					}
					else {
						sphere.metallic = false;
						sphere.transparent = true;
						sphere.roughness = 0.25f * row;
					}
                }

                if (!overlap)
				    spheres.push_back(sphere);
			}

            // Create array for the sphere centers and the radii
            float3* sphere_centers = new float3[NUM_SPHERES];
            float* sphere_radii = new float[NUM_SPHERES];

            // Copy the sphere data into the arrays
            for (int i = 0; i < NUM_SPHERES; ++i) {
                sphere_centers[i] = spheres[i].center;
                sphere_radii[i] = spheres[i].radius;
            }

            // Create an array of indices for the materials of the spheres
            std::vector<uint32_t> g_mat_indices;
            for (int i = 0; i < NUM_SPHERES; ++i) {
				g_mat_indices.push_back(i);
			}

            // Allocate device memory for the array of material indices and copy the data from host to device
            CUdeviceptr  d_mat_indices = 0;
            const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_mat_indices),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
            ));

            // Allocate device memory for the array of spheres' center vertices and copy the data from host to device
            CUdeviceptr d_vertex_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer), spheres.size() * sizeof(float3)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buffer), sphere_centers,
                				spheres.size() * sizeof(float3), cudaMemcpyHostToDevice));

            // Allocate device memory for the array of spheres' radii and copy the data from host to device
            CUdeviceptr d_radius_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buffer), spheres.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buffer), sphere_radii,
                								spheres.size() * sizeof(float), cudaMemcpyHostToDevice));

            // Configure the build input to describe the spheres with the provided vertex and radius buffers
            OptixBuildInput sphere_input = {};
            sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
            sphere_input.sphereArray.numVertices = spheres.size();
            sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;
            uint32_t sphere_input_flags[NUM_SPHERES];
            for (int i = 0; i < NUM_SPHERES; ++i) {
				sphere_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
			}
            sphere_input.sphereArray.flags = sphere_input_flags;
            sphere_input.sphereArray.numSbtRecords = NUM_SPHERES;
            sphere_input.sphereArray.sbtIndexOffsetBuffer = d_mat_indices; // Buffer with the material indices
            sphere_input.sphereArray.sbtIndexOffsetSizeInBytes = sizeof(unsigned int); // Size of the material indices
            sphere_input.sphereArray.sbtIndexOffsetStrideInBytes = sizeof(unsigned int); // Stride of the material indices

            // Compute the memory required for the GAS
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));

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

            // Build the actual GAS and retrieve the compacted size
            OPTIX_CHECK(optixAccelBuild(context,
                0,  // CUDA stream
                &accel_options, &sphere_input,
                1,  // Number of build inputs
                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                &emitProperty,  // Emitted property list
                1               // Number of emitted properties
            ));

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

            // Assign the buffer for GAS output
            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

            // Free temporary buffers
            CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
            CUDA_CHECK(cudaFree((void*)d_vertex_buffer));
            CUDA_CHECK(cudaFree((void*)d_radius_buffer));

            // If compacted size is smaller, create a buffer of that size and do the compaction
            size_t compacted_gas_size;
            CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

            // If the GAS is not yet compacted, do so now
            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));
                OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));
                CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
            } else {
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
        OptixModule sphere_module = nullptr;
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
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

            // Load CUDA source from file and create the module containing the compiled CUDA functions (shaders)
            size_t      inputSize = 0;
            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixSphere.cu", inputSize);
            OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, input,
                inputSize, LOG, &LOG_SIZE, &module));

            // Built-in intersection module creation for spheres:
            // Apart from user-defined intersection programs, OptiX also provides built-in intersection programs for certain primitives like spheres and triangles.
            OptixBuiltinISOptions builtin_is_options = {};
            builtin_is_options.usesMotionBlur = false;
            builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
            OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options,
                &builtin_is_options, &sphere_module));
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
            hitgroup_prog_group_desc.hitgroup.moduleIS = sphere_module; // Use the built-in sphere module for intersection
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

            // Miss record setup - contains the background color for rays that miss geometry
            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.35f, 0.762f, 0.95f };
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
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_records), hitgroup_record_size * NUM_SPHERES));

            // Populate the SBT records with each sphere's color data
            HitGroupSbtRecord* hg_sbts = new HitGroupSbtRecord[NUM_SPHERES];

            // Populate the hit group records with their associated data
            for (size_t i = 0; i < NUM_SPHERES; ++i)
            {
                // Pack the hitgroup_prog_group's shader identifier into each hit group record's header.
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbts[i]));

                // Fill each of your SBT records with the appropriate color
                hg_sbts[i].data.diffuse_color = spheres[i].color;  // Assuming 'spheres' is the vector of SphereData you allocated earlier.
				hg_sbts[i].data.emission_color = spheres[i].color * spheres[i].emission;
                hg_sbts[i].data.specular = spheres[i].specular;
                hg_sbts[i].data.roughness = spheres[i].roughness;
                hg_sbts[i].data.metallic = spheres[i].metallic;
                hg_sbts[i].data.transparent = spheres[i].transparent;
            }

            // Copy the hit group SBT records to the device
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hitgroup_records),
                hg_sbts,
                hitgroup_record_size * NUM_SPHERES,
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
            sbt.hitgroupRecordCount = NUM_SPHERES;
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
            params.image_width* params.image_height * sizeof(float4)
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

            // Free the GAS output buffer
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

            // Destroy the OptiX objects created
            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));
            OPTIX_CHECK(optixModuleDestroy(sphere_module));

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