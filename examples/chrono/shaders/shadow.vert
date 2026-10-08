#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_position;

void main() {
  gl_Position = frame.light_view_proj * instances[gl_InstanceIndex].model * vec4(in_position, 1.0);
}
