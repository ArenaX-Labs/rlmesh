// The Vulkan renderer against the CPU ray tracer, on scenes the env never
// draws: enough spheres and capsules to grow the instance buffer three times past
// its initial 256 instances, several frame sizes, and frame sizes it must
// refuse without leaving anything broken behind. Exits non-zero on a failure.
//
//   chrono_reach_render_test
//
// Run it under VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation to have the
// validation layers check every call, and with VK_DRIVER_FILES pointing at
// lavapipe to test on the CPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "raytracer.h"
#include "renderer.h"

namespace {

using scene::Vec3;

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

// The robot-sized scene the env draws: a few links and joints.
scene::Scene small_scene() {
  scene::Scene s;
  const Vec3 steel(0.93, 0.55, 0.12), joint(0.18, 0.19, 0.22);
  s.capsules.push_back({{0, 0.0, 0}, {0, 0.03, 0}, 0.11, joint});
  s.capsules.push_back({{0, 0.03, 0}, {0, 0.35, 0}, 0.06, steel});
  s.capsules.push_back({{0, 0.35, 0}, {0.3, 0.45, 0.05}, 0.045, steel});
  s.spheres.push_back({{0, 0.35, 0}, 0.07, joint});
  s.spheres.push_back({{0.3, 0.45, 0.05}, 0.05, joint});
  s.spheres.push_back({{0.35, 0.25, 0.1}, 0.03, {0.9, 0.12, 0.12}});
  return s;
}

// A grid of `n` spheres on the floor around the camera's target.
scene::Scene many_spheres(int n) {
  scene::Scene s;
  const int side = static_cast<int>(std::ceil(std::sqrt(n)));
  for (int i = 0; i < n; ++i) {
    const double x = -0.3 + 0.9 * (i % side) / side, z = -0.5 + 0.9 * (i / side) / side;
    const double r = 0.012 + 0.006 * (i % 3);
    s.spheres.push_back({{x, r + 0.1 * (i % 5) / 5, z}, r, {0.2 + 0.6 * (i % 4) / 3, 0.5, 0.8}});
  }
  return s;
}

// `n` capsules leaning in a fan: 3 instances each (two end spheres and a
// cylinder).
scene::Scene many_capsules(int n) {
  scene::Scene s;
  const int side = static_cast<int>(std::ceil(std::sqrt(n)));
  for (int i = 0; i < n; ++i) {
    const double x = -0.3 + 0.9 * (i % side) / side, z = -0.5 + 0.9 * (i / side) / side;
    const double lean = 0.03 * std::sin(i * 0.7);
    s.capsules.push_back(
        {{x, 0.01, z}, {x + lean, 0.08 + 0.04 * (i % 4), z - lean}, 0.008, {0.93, 0.55, 0.12}});
  }
  return s;
}

// The share of pixels where any channel differs by more than 48 of 255:
// silhouettes and shadow edges, where the tessellated meshes and the shadow
// map's texels part from the ray tracer's analytic shapes.
double mismatch(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 1.0;
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); i += 3) {
    for (size_t c = 0; c < 3; ++c) {
      if (std::abs(int(a[i + c]) - int(b[i + c])) > 48) {
        ++bad;
        break;
      }
    }
  }
  return static_cast<double>(bad) / (a.size() / 3);
}

// Renders with both backends and compares them.
std::vector<uint8_t> compare(chrono_reach::Renderer& vulkan, chrono_reach::Renderer& reference,
                             const std::string& name, const scene::Scene& s, int width, int height,
                             double tolerance) {
  const std::string what = name + " at " + std::to_string(width) + "x" + std::to_string(height);
  std::vector<uint8_t> image;
  try {
    image = vulkan.render(s, scene::Camera{}, width, height);
  } catch (const std::exception& error) {
    expect(false, what + ": " + error.what());
    return image;
  }
  expect(image.size() == static_cast<size_t>(width) * height * 3, what + ": RGB8 frame size");
  const double share = mismatch(image, reference.render(s, scene::Camera{}, width, height));
  char line[64];
  std::snprintf(line, sizeof(line), ": %.2f%% of pixels differ (<= %.1f%%)", 100 * share,
                100 * tolerance);
  expect(share <= tolerance, what + line);
  return image;
}

bool throws(chrono_reach::Renderer& renderer, int width, int height) {
  try {
    renderer.render(small_scene(), scene::Camera{}, width, height);
  } catch (const std::exception& error) {
    std::printf("     (%dx%d: %s)\n", width, height, error.what());
    return true;
  }
  return false;
}

}  // namespace

int main() {
  std::unique_ptr<chrono_reach::Renderer> vulkan;
  try {
    vulkan = chrono_reach::make_vulkan_renderer();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "no Vulkan renderer: %s\n", error.what());
    return 1;
  }
  const auto reference = chrono_reach::make_raytracer();
  std::printf("%s\n", vulkan->describe().c_str());

  const scene::Scene small = small_scene();
  const std::vector<uint8_t> first =
      compare(*vulkan, *reference, "small scene", small, 256, 256, 0.005);
  // 1 floor + 300 spheres: past the initial 256 instances.
  compare(*vulkan, *reference, "300 spheres", many_spheres(300), 96, 96, 0.04);
  // 1 floor + 1200 capsule instances: past the 602 the first growth made room for.
  compare(*vulkan, *reference, "400 capsules", many_capsules(400), 160, 120, 0.04);
  // 3001 instances, past the 2402 of the second growth, in a frame taller than
  // it is wide.
  compare(*vulkan, *reference, "3000 spheres", many_spheres(3000), 120, 200, 0.05);

  // Back to the small scene in its cached target: the same pixels as before
  // the buffer grew.
  expect(vulkan->render(small, scene::Camera{}, 256, 256) == first,
         "small scene after growth matches its first frame");

  // Sizes the renderer must refuse, twice each so a failure cannot leave a
  // broken target cached, and then a good frame still renders.
  expect(throws(*vulkan, 0, 64), "a zero-width frame is refused");
  expect(throws(*vulkan, 1 << 20, 1), "an oversized frame is refused");
  expect(throws(*vulkan, 1 << 20, 1), "an oversized frame is refused again");
  expect(vulkan->render(small, scene::Camera{}, 256, 256) == first,
         "small scene after refused frames matches its first frame");
  compare(*vulkan, *reference, "small scene", small, 64, 48, 0.01);

  vulkan.reset();
  std::printf("%s\n", failures == 0 ? "all passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}
