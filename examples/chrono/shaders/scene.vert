#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) flat out vec4 color;

void main() {
  Instance instance = instances[gl_InstanceIndex];
  vec4 position = instance.model * vec4(in_position, 1.0);
  world_position = position.xyz;
  world_normal = transpose(inverse(mat3(instance.model))) * in_normal;
  color = instance.color;
  gl_Position = frame.view_proj * position;
}
