#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec2 ndc;
layout(location = 0) out vec4 out_color;

// A soft vertical gradient. Vulkan's NDC y points down, the camera's up.
void main() {
  vec3 rd = normalize(frame.forward.xyz + frame.right.xyz * (ndc.x * frame.right.w) +
                      frame.up.xyz * (-ndc.y * frame.up.w));
  float k = 0.5 * (rd.y + 1.0);
  out_color = vec4(mix(kHorizon, kZenith, k), 1.0);
}
