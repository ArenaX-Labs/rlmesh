// The env's camera backends: each turns a scene (scene.h) into an RGB8 image.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "scene.h"

namespace chrono_reach {

class Renderer {
 public:
  virtual ~Renderer() = default;
  // A human-readable name, e.g. the GPU a Vulkan renderer runs on.
  virtual std::string describe() const = 0;
  // Row-major RGB8, width x height x 3. Throws std::runtime_error on failure.
  virtual std::vector<uint8_t> render(const scene::Scene& scene, const scene::Camera& camera,
                                      int width, int height) = 0;
};

// The CPU ray tracer (raytracer.h): always available.
std::unique_ptr<Renderer> make_raytracer();

// An offscreen Vulkan rasterizer (vulkan_renderer.cpp), when built with
// CHRONO_REACH_VULKAN. Throws std::runtime_error when no Vulkan device works,
// or when the binary was built without it.
std::unique_ptr<Renderer> make_vulkan_renderer();

}  // namespace chrono_reach
