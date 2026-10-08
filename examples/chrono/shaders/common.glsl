// Shared by the Vulkan renderer's shaders (vulkan_renderer.cpp mirrors these
// layouts). Shading matches the CPU ray tracer in raytracer.h.

layout(set = 0, binding = 0, std140) uniform Frame {
  mat4 view_proj;
  mat4 light_view_proj;
  vec4 eye;
  vec4 light;    // direction towards the light, normalized
  vec4 forward;  // camera basis for the sky; right.w and up.w are the half extents
  vec4 right;
  vec4 up;
}
frame;

struct Instance {
  mat4 model;
  vec4 color;  // rgb albedo; w = 1 for the floor
};

layout(set = 0, binding = 1, std430) readonly buffer Instances { Instance instances[]; };

const vec3 kHorizon = vec3(0.78, 0.84, 0.92);
const vec3 kZenith = vec3(0.42, 0.55, 0.75);
