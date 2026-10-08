// A tiny CPU ray tracer for the env's camera (see scene.h): hard shadows from
// one directional light. No GPU, no display, no dependencies, so the same
// binary renders headless in a container.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "scene.h"

namespace raytracer {

using scene::Camera;
using scene::Capsule;
using scene::cross;
using scene::dot;
using scene::normalize;
using scene::Scene;
using scene::Sphere;
using scene::Vec3;

namespace detail {

constexpr double kMiss = 1e30;

inline double hit_sphere(const Vec3& ro, const Vec3& rd, const Sphere& s, Vec3* normal) {
  const Vec3 oc = ro - s.center;
  const double b = dot(oc, rd);
  const double c = dot(oc, oc) - s.radius * s.radius;
  const double h = b * b - c;
  if (h < 0) return kMiss;
  const double t = -b - std::sqrt(h);
  if (t <= 1e-6) return kMiss;
  if (normal) *normal = normalize(ro + rd * t - s.center);
  return t;
}

// Inigo Quilez's analytic ray-capsule intersection.
inline double hit_capsule(const Vec3& ro, const Vec3& rd, const Capsule& c, Vec3* normal) {
  const Vec3 axis = c.b - c.a;
  const Vec3 oa = ro - c.a;
  const double axis2 = dot(axis, axis);
  const double axis_rd = dot(axis, rd);
  const double axis_oa = dot(axis, oa);
  const double rdoa = dot(rd, oa);
  const double oaoa = dot(oa, oa);
  const double ra = c.radius;
  double a = axis2 - axis_rd * axis_rd;
  double b = axis2 * rdoa - axis_oa * axis_rd;
  double k = axis2 * oaoa - axis_oa * axis_oa - ra * ra * axis2;
  double h = b * b - a * k;
  double t = kMiss;
  if (h >= 0.0 && a > 1e-12) {
    const double body = (-b - std::sqrt(h)) / a;
    const double y = axis_oa + body * axis_rd;
    if (y > 0.0 && y < axis2 && body > 1e-6) {
      t = body;
    } else {
      // The end caps.
      const Vec3 oc = (y <= 0.0) ? oa : ro - c.b;
      b = dot(rd, oc);
      k = dot(oc, oc) - ra * ra;
      h = b * b - k;
      if (h > 0.0) {
        const double cap = -b - std::sqrt(h);
        if (cap > 1e-6) t = cap;
      }
    }
  } else {
    // Ray parallel to the axis: the caps alone.
    for (const Vec3& end : {c.a, c.b}) {
      Sphere cap{end, ra, c.color};
      t = std::min(t, hit_sphere(ro, rd, cap, nullptr));
    }
  }
  if (t >= kMiss) return kMiss;
  if (normal) {
    const Vec3 p = ro + rd * t;
    const Vec3 pa = p - c.a;
    const double hh = std::clamp(dot(pa, axis) / axis2, 0.0, 1.0);
    *normal = normalize(pa - axis * hh);
  }
  return t;
}

struct Hit {
  double t = kMiss;
  Vec3 normal;
  Vec3 color;
  bool floor = false;
};

inline Hit trace(const Scene& scene, const Vec3& ro, const Vec3& rd) {
  Hit best;
  Vec3 normal;
  for (const Sphere& s : scene.spheres) {
    const double t = hit_sphere(ro, rd, s, &normal);
    if (t < best.t) best = {t, normal, s.color, false};
  }
  for (const Capsule& c : scene.capsules) {
    const double t = hit_capsule(ro, rd, c, &normal);
    if (t < best.t) best = {t, normal, c.color, false};
  }
  if (rd.y < -1e-9) {
    const double t = -ro.y / rd.y;
    if (t > 1e-6 && t < best.t) best = {t, Vec3(0, 1, 0), Vec3(), true};
  }
  return best;
}

inline bool occluded(const Scene& scene, const Vec3& p, const Vec3& light) {
  for (const Sphere& s : scene.spheres) {
    if (hit_sphere(p, light, s, nullptr) < kMiss) return true;
  }
  for (const Capsule& c : scene.capsules) {
    if (hit_capsule(p, light, c, nullptr) < kMiss) return true;
  }
  return false;
}

inline uint8_t to_byte(double v) {
  return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
}

}  // namespace detail

// Render `scene` to a row-major RGB8 image of `width` x `height`.
inline std::vector<uint8_t> render(const Scene& scene, const Camera& cam, int width, int height) {
  using namespace detail;
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3);
  const Vec3 forward = normalize(cam.target - cam.eye);
  const Vec3 right = normalize(cross(forward, Vec3(0, 1, 0)));
  const Vec3 up = cross(right, forward);
  const double half = std::tan(cam.fov_deg * M_PI / 360.0);
  const double aspect = static_cast<double>(width) / height;
  const Vec3 light = normalize(scene::light_direction());

  for (int py = 0; py < height; ++py) {
    for (int px = 0; px < width; ++px) {
      const double u = (2.0 * (px + 0.5) / width - 1.0) * half * aspect;
      const double v = (1.0 - 2.0 * (py + 0.5) / height) * half;
      const Vec3 rd = normalize(forward + right * u + up * v);
      const Hit hit = trace(scene, cam.eye, rd);

      Vec3 color;
      if (hit.t >= kMiss) {
        // Sky: a soft vertical gradient.
        const double k = 0.5 * (rd.y + 1.0);
        color = Vec3(0.78, 0.84, 0.92) * (1.0 - k) + Vec3(0.42, 0.55, 0.75) * k;
      } else {
        const Vec3 p = cam.eye + rd * hit.t;
        Vec3 albedo = hit.color;
        if (hit.floor) {
          const bool even =
              (static_cast<int>(std::floor(p.x * 5)) + static_cast<int>(std::floor(p.z * 5))) % 2 ==
              0;
          albedo = even ? Vec3(0.82, 0.82, 0.80) : Vec3(0.62, 0.63, 0.62);
          // Fade the floor into the horizon.
          const double fade = std::clamp(hit.t / 6.0, 0.0, 1.0);
          albedo = albedo * (1.0 - fade) + Vec3(0.78, 0.84, 0.92) * fade;
        }
        const bool shadow = occluded(scene, p + hit.normal * 1e-4, light);
        const double diffuse = shadow ? 0.0 : std::max(0.0, dot(hit.normal, light));
        double specular = 0.0;
        if (!shadow && !hit.floor) {
          const Vec3 half_vec = normalize(light - rd);
          specular = std::pow(std::max(0.0, dot(hit.normal, half_vec)), 40.0) * 0.35;
        }
        color = albedo * (0.32 + 0.68 * diffuse) + Vec3(1, 1, 1) * specular;
      }
      uint8_t* out = &pixels[(static_cast<size_t>(py) * width + px) * 3];
      out[0] = to_byte(color.x);
      out[1] = to_byte(color.y);
      out[2] = to_byte(color.z);
    }
  }
  return pixels;
}

}  // namespace raytracer
