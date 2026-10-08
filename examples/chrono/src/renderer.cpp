#include "renderer.h"

#include <stdexcept>

#include "raytracer.h"

namespace chrono_reach {

namespace {

class RayTracer : public Renderer {
 public:
  std::string describe() const override { return "raytrace (CPU)"; }
  std::vector<uint8_t> render(const scene::Scene& scene, const scene::Camera& camera, int width,
                              int height) override {
    return raytracer::render(scene, camera, width, height);
  }
};

}  // namespace

std::unique_ptr<Renderer> make_raytracer() { return std::make_unique<RayTracer>(); }

#ifndef CHRONO_REACH_VULKAN
std::unique_ptr<Renderer> make_vulkan_renderer() {
  throw std::runtime_error(
      "this build has no Vulkan renderer: configure with -DCHRONO_REACH_VULKAN=ON");
}
#endif

}  // namespace chrono_reach
