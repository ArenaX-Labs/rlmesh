// The camera's scene, shared by every renderer: spheres and capsules over a
// checkered floor, lit by one directional light. The CPU ray tracer and the
// Vulkan renderer both draw it, with the same camera and shading.
//
// World convention matches the Chrono scene: Y is up, the floor is y = 0.
#pragma once

#include <cmath>
#include <vector>

namespace scene {

struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
};

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(const Vec3& v) {
  const double n = std::sqrt(dot(v, v));
  return n > 0 ? v * (1.0 / n) : v;
}

struct Sphere {
  Vec3 center;
  double radius;
  Vec3 color;
};

// A capsule: the segment a-b swept by `radius` (a robot link).
struct Capsule {
  Vec3 a, b;
  double radius;
  Vec3 color;
};

struct Scene {
  std::vector<Sphere> spheres;
  std::vector<Capsule> capsules;
};

struct Camera {
  Vec3 eye{0.95, 0.72, 1.0};
  Vec3 target{0.26, 0.2, 0.02};
  double fov_deg = 42;
};

// Direction towards the light (normalized by the renderers).
inline Vec3 light_direction() { return Vec3(0.5, 1.0, 0.35); }

}  // namespace scene
