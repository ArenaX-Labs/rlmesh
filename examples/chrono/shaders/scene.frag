#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(set = 0, binding = 2) uniform sampler2DShadow shadow_map;

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) flat in vec4 color;

layout(location = 0) out vec4 out_color;

// 1 when `p` sees the light, 0 in shadow.
float lit(vec3 p) {
  vec4 clip = frame.light_view_proj * vec4(p, 1.0);
  vec3 ndc = clip.xyz / clip.w;
  vec2 uv = ndc.xy * 0.5 + 0.5;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || ndc.z > 1.0) return 1.0;
  return texture(shadow_map, vec3(uv, ndc.z));
}

void main() {
  vec3 n = normalize(world_normal);
  vec3 rd = normalize(world_position - frame.eye.xyz);
  vec3 light = frame.light.xyz;
  bool is_floor = color.w > 0.5;

  vec3 albedo = color.rgb;
  if (is_floor) {
    ivec2 cell = ivec2(floor(world_position.xz * 5.0));
    albedo = ((cell.x + cell.y) & 1) == 0 ? vec3(0.82, 0.82, 0.80) : vec3(0.62, 0.63, 0.62);
    // Fade the floor into the horizon.
    float fade = clamp(length(world_position - frame.eye.xyz) / 6.0, 0.0, 1.0);
    albedo = mix(albedo, kHorizon, fade);
  }
  float facing = dot(n, light);
  float visible = facing > 0.0 ? lit(world_position + n * 0.004) : 0.0;
  float diffuse = max(facing, 0.0) * visible;
  float specular = 0.0;
  if (!is_floor && visible > 0.0) {
    vec3 half_vec = normalize(light - rd);
    specular = pow(max(dot(n, half_vec), 0.0), 40.0) * 0.35;
  }
  out_color = vec4(clamp(albedo * (0.32 + 0.68 * diffuse) + vec3(specular), 0.0, 1.0), 1.0);
}
