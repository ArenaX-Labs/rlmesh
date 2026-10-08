// An offscreen Vulkan rasterizer for the env's camera: no window, no surface,
// no display server. It draws the same scene as the CPU ray tracer (scene.h)
// from instanced sphere and cylinder meshes over a floor plane, with a shadow
// map standing in for the ray tracer's shadow rays, then copies the frame back
// to host memory. It runs on any Vulkan 1.1 device with a graphics queue: a
// GPU, or Mesa's lavapipe on the CPU (VK_DRIVER_FILES=.../lvp_icd.*.json).
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "renderer.h"

// SPIR-V compiled from ../shaders at build time (glslangValidator --vn).
#include "scene_frag.spv.h"
#include "scene_vert.spv.h"
#include "shadow_vert.spv.h"
#include "sky_frag.spv.h"
#include "sky_vert.spv.h"

namespace chrono_reach {

namespace {

using scene::Vec3;

void check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(result) +
                             ")");
  }
}

// Column-major, as GLSL reads it.
struct Mat4 {
  std::array<float, 16> m{};
  float& operator()(int row, int col) { return m[col * 4 + row]; }
  float operator()(int row, int col) const { return m[col * 4 + row]; }
};

Mat4 operator*(const Mat4& a, const Mat4& b) {
  Mat4 out;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      float sum = 0;
      for (int k = 0; k < 4; ++k) sum += a(row, k) * b(k, col);
      out(row, col) = sum;
    }
  }
  return out;
}

// A model matrix from three (scaled) basis columns and a translation.
Mat4 basis(const Vec3& x, const Vec3& y, const Vec3& z, const Vec3& origin) {
  Mat4 out;
  const Vec3 cols[4] = {x, y, z, origin};
  for (int col = 0; col < 4; ++col) {
    out(0, col) = static_cast<float>(cols[col].x);
    out(1, col) = static_cast<float>(cols[col].y);
    out(2, col) = static_cast<float>(cols[col].z);
  }
  out(3, 3) = 1;
  return out;
}

Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& world_up) {
  const Vec3 f = scene::normalize(target - eye);
  const Vec3 s = scene::normalize(scene::cross(f, world_up));
  const Vec3 u = scene::cross(s, f);
  Mat4 out;
  const Vec3 rows[3] = {s, u, f * -1.0};
  for (int row = 0; row < 3; ++row) {
    out(row, 0) = static_cast<float>(rows[row].x);
    out(row, 1) = static_cast<float>(rows[row].y);
    out(row, 2) = static_cast<float>(rows[row].z);
    out(row, 3) = static_cast<float>(-scene::dot(rows[row], eye));
  }
  out(3, 3) = 1;
  return out;
}

// Vulkan clip space: y down, depth in [0, 1].
Mat4 perspective(double fovy_rad, double aspect, double near, double far) {
  const double t = std::tan(fovy_rad / 2);
  Mat4 out;
  out(0, 0) = static_cast<float>(1 / (aspect * t));
  out(1, 1) = static_cast<float>(-1 / t);
  out(2, 2) = static_cast<float>(far / (near - far));
  out(2, 3) = static_cast<float>(near * far / (near - far));
  out(3, 2) = -1;
  return out;
}

Mat4 orthographic(double half, double near, double far) {
  Mat4 out;
  out(0, 0) = static_cast<float>(1 / half);
  out(1, 1) = static_cast<float>(1 / half);
  out(2, 2) = static_cast<float>(-1 / (far - near));
  out(2, 3) = static_cast<float>(-near / (far - near));
  out(3, 3) = 1;
  return out;
}

void store(float* out, const Vec3& v, double w) {
  out[0] = static_cast<float>(v.x);
  out[1] = static_cast<float>(v.y);
  out[2] = static_cast<float>(v.z);
  out[3] = static_cast<float>(w);
}

// The shaders' layouts (shaders/common.glsl): std140 Frame, std430 Instance.
struct FrameUniforms {
  float view_proj[16];
  float light_view_proj[16];
  float eye[4];
  float light[4];
  float forward[4];
  float right[4];
  float up[4];
};

struct Instance {
  float model[16];
  float color[4];
};

Instance instance(const Mat4& model, const Vec3& color, double kind = 0) {
  Instance out;
  std::memcpy(out.model, model.m.data(), sizeof(out.model));
  store(out.color, color, kind);
  return out;
}

struct Vertex {
  float position[3];
  float normal[3];
};

struct Mesh {
  uint32_t first_index = 0;
  uint32_t index_count = 0;
  int32_t vertex_offset = 0;
};

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kShadowSize = 2048;
constexpr double kFloorExtent = 200;  // metres: the floor reaches the horizon

class VulkanRenderer : public Renderer {
 public:
  VulkanRenderer() {
    try {
      init();
    } catch (...) {
      destroy();
      throw;
    }
  }
  ~VulkanRenderer() override { destroy(); }
  VulkanRenderer(const VulkanRenderer&) = delete;
  VulkanRenderer& operator=(const VulkanRenderer&) = delete;

  std::string describe() const override { return description_; }

  std::vector<uint8_t> render(const scene::Scene& scene, const scene::Camera& camera, int width,
                              int height) override;

 private:
  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool coherent = true;
  };
  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
  };
  // An offscreen color + depth target and its readback buffer, per frame size.
  struct Target {
    Image color, depth;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    Buffer readback;
  };

  void init();
  void destroy();
  void pick_device();
  void create_render_passes();
  void create_shadow_map();
  void create_descriptors();
  void create_pipelines();
  void create_meshes();
  void write_descriptors();

  uint32_t memory_type(uint32_t bits, std::initializer_list<VkMemoryPropertyFlags> choices,
                       VkMemoryPropertyFlags* chosen) const;
  Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool readback = false);
  void destroy_buffer(Buffer& buffer);
  Image make_image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                   VkImageAspectFlags aspect);
  void destroy_image(Image& image);
  VkShaderModule shader(const uint32_t* code, size_t bytes);
  VkPipeline pipeline(VkShaderModule vert, VkShaderModule frag, VkRenderPass pass, bool vertices,
                      bool depth_write, VkCompareOp depth_compare, bool depth_bias);
  Target& target(int width, int height);

  std::string description_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_{};
  uint32_t queue_family_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer commands_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;

  VkRenderPass shadow_pass_ = VK_NULL_HANDLE;
  VkRenderPass main_pass_ = VK_NULL_HANDLE;
  Image shadow_map_;
  VkFramebuffer shadow_framebuffer_ = VK_NULL_HANDLE;
  VkSampler shadow_sampler_ = VK_NULL_HANDLE;

  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkPipeline scene_pipeline_ = VK_NULL_HANDLE;
  VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
  VkPipeline sky_pipeline_ = VK_NULL_HANDLE;

  Buffer vertices_, indices_, uniforms_, instances_;
  Mesh sphere_, cylinder_, floor_;
  std::map<std::pair<int, int>, Target> targets_;
};

void VulkanRenderer::init() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "chrono_reach_env";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pApplicationInfo = &app;
  check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

  pick_device();

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = queue_family_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  check(vkCreateDevice(physical_, &device_info, nullptr, &device_), "vkCreateDevice");
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_;
  check(vkCreateCommandPool(device_, &pool_info, nullptr, &pool_), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool_;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(device_, &alloc, &commands_), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(vkCreateFence(device_, &fence_info, nullptr, &fence_), "vkCreateFence");

  create_render_passes();
  create_shadow_map();
  create_descriptors();
  create_pipelines();
  create_meshes();
  write_descriptors();
}

void VulkanRenderer::pick_device() {
  uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
  std::vector<VkPhysicalDevice> devices(count);
  check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
        "vkEnumeratePhysicalDevices");

  auto rank = [](VkPhysicalDeviceType type) {
    switch (type) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 0;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 1;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 2;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return 3;
      default:
        return 4;
    }
  };
  auto supports = [](VkPhysicalDevice device, VkFormat format, VkFormatFeatureFlags features) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(device, format, &props);
    return (props.optimalTilingFeatures & features) == features;
  };

  int best_rank = 1 << 30;
  VkPhysicalDeviceProperties best_props{};
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    if (props.apiVersion < VK_API_VERSION_1_1) continue;
    if (!supports(device, kColorFormat,
                  VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ||
        !supports(
            device, kDepthFormat,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
      continue;
    }
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &families, nullptr);
    std::vector<VkQueueFamilyProperties> family_props(families);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &families, family_props.data());
    for (uint32_t i = 0; i < families; ++i) {
      if ((family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          rank(props.deviceType) < best_rank) {
        best_rank = rank(props.deviceType);
        physical_ = device;
        queue_family_ = i;
        best_props = props;
        break;
      }
    }
  }
  if (physical_ == VK_NULL_HANDLE) {
    throw std::runtime_error("no Vulkan 1.1 device with a graphics queue (" +
                             std::to_string(count) + " devices found)");
  }
  vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
  const char* kinds[] = {"discrete GPU", "integrated GPU", "virtual GPU", "CPU", "other"};
  description_ = std::string("vulkan (") + best_props.deviceName + ", " + kinds[best_rank] + ")";
}

void VulkanRenderer::create_render_passes() {
  // Shadow pass: depth only, left readable by the main pass's fragment shader.
  VkAttachmentDescription depth{};
  depth.format = kDepthFormat;
  depth.samples = VK_SAMPLE_COUNT_1_BIT;
  depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  VkAttachmentReference depth_ref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription shadow_subpass{};
  shadow_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  shadow_subpass.pDepthStencilAttachment = &depth_ref;
  const VkPipelineStageFlags depth_stages =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  std::array<VkSubpassDependency, 2> shadow_deps{};
  shadow_deps[0] = {VK_SUBPASS_EXTERNAL,
                    0,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    depth_stages,
                    0,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    0};
  shadow_deps[1] = {0,
                    VK_SUBPASS_EXTERNAL,
                    depth_stages,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    0};
  VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  info.attachmentCount = 1;
  info.pAttachments = &depth;
  info.subpassCount = 1;
  info.pSubpasses = &shadow_subpass;
  info.dependencyCount = static_cast<uint32_t>(shadow_deps.size());
  info.pDependencies = shadow_deps.data();
  check(vkCreateRenderPass(device_, &info, nullptr, &shadow_pass_), "vkCreateRenderPass(shadow)");

  // Main pass: color + depth, the color left ready to copy back to the host.
  std::array<VkAttachmentDescription, 2> attachments{};
  attachments[0].format = kColorFormat;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  attachments[1] = depth;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference main_depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription main_subpass{};
  main_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  main_subpass.colorAttachmentCount = 1;
  main_subpass.pColorAttachments = &color_ref;
  main_subpass.pDepthStencilAttachment = &main_depth_ref;
  std::array<VkSubpassDependency, 2> main_deps{};
  main_deps[0] = {
      VK_SUBPASS_EXTERNAL,
      0,
      VK_PIPELINE_STAGE_TRANSFER_BIT | depth_stages,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | depth_stages,
      0,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      0};
  main_deps[1] = {0,
                  VK_SUBPASS_EXTERNAL,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT,
                  0};
  info.attachmentCount = static_cast<uint32_t>(attachments.size());
  info.pAttachments = attachments.data();
  info.pSubpasses = &main_subpass;
  info.dependencyCount = static_cast<uint32_t>(main_deps.size());
  info.pDependencies = main_deps.data();
  check(vkCreateRenderPass(device_, &info, nullptr, &main_pass_), "vkCreateRenderPass(main)");
}

void VulkanRenderer::create_shadow_map() {
  shadow_map_ = make_image(kShadowSize, kShadowSize, kDepthFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
  VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fb.renderPass = shadow_pass_;
  fb.attachmentCount = 1;
  fb.pAttachments = &shadow_map_.view;
  fb.width = kShadowSize;
  fb.height = kShadowSize;
  fb.layers = 1;
  check(vkCreateFramebuffer(device_, &fb, nullptr, &shadow_framebuffer_),
        "vkCreateFramebuffer(shadow)");

  // Hard shadows, like the ray tracer's shadow rays: one nearest compare.
  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler.magFilter = VK_FILTER_NEAREST;
  sampler.minFilter = VK_FILTER_NEAREST;
  sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  sampler.compareEnable = VK_TRUE;
  sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  check(vkCreateSampler(device_, &sampler, nullptr, &shadow_sampler_), "vkCreateSampler");
}

void VulkanRenderer::create_descriptors() {
  std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                 nullptr};
  VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout.bindingCount = static_cast<uint32_t>(bindings.size());
  layout.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(device_, &layout, nullptr, &set_layout_),
        "vkCreateDescriptorSetLayout");
  VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &set_layout_;
  check(vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &pipeline_layout_),
        "vkCreatePipelineLayout");

  std::array<VkDescriptorPoolSize, 3> sizes{{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}};
  VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool.maxSets = 1;
  pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
  pool.pPoolSizes = sizes.data();
  check(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = descriptor_pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &set_layout_;
  check(vkAllocateDescriptorSets(device_, &alloc, &set_), "vkAllocateDescriptorSets");

  uniforms_ = make_buffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  instances_ = make_buffer(256 * sizeof(Instance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void VulkanRenderer::write_descriptors() {
  VkDescriptorBufferInfo frame{uniforms_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo instances{instances_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo shadow{shadow_sampler_, shadow_map_.view,
                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
  std::array<VkWriteDescriptorSet, 3> writes{};
  for (uint32_t i = 0; i < writes.size(); ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set_;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
  }
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo = &frame;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &instances;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[2].pImageInfo = &shadow;
  vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkShaderModule VulkanRenderer::shader(const uint32_t* code, size_t bytes) {
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = bytes;
  info.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device_, &info, nullptr, &module), "vkCreateShaderModule");
  return module;
}

VkPipeline VulkanRenderer::pipeline(VkShaderModule vert, VkShaderModule frag, VkRenderPass pass,
                                    bool vertices, bool depth_write, VkCompareOp depth_compare,
                                    bool depth_bias) {
  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               nullptr,
               0,
               VK_SHADER_STAGE_VERTEX_BIT,
               vert,
               "main",
               nullptr};
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               nullptr,
               0,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               frag,
               "main",
               nullptr};

  VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  std::array<VkVertexInputAttributeDescription, 2> attributes{
      {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
       {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}}};
  VkPipelineVertexInputStateCreateInfo input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  if (vertices) {
    input.vertexBindingDescriptionCount = 1;
    input.pVertexBindingDescriptions = &binding;
    // The shadow pass reads positions only.
    input.vertexAttributeDescriptionCount = frag != VK_NULL_HANDLE ? 2 : 1;
    input.pVertexAttributeDescriptions = attributes.data();
  }
  VkPipelineInputAssemblyStateCreateInfo assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;
  raster.depthBiasEnable = depth_bias ? VK_TRUE : VK_FALSE;
  raster.depthBiasConstantFactor = 1.0f;
  raster.depthBiasSlopeFactor = 1.5f;
  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo depth{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth.depthTestEnable = VK_TRUE;
  depth.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
  depth.depthCompareOp = depth_compare;
  VkPipelineColorBlendAttachmentState blend{};
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color.attachmentCount = frag != VK_NULL_HANDLE ? 1 : 0;
  color.pAttachments = &blend;
  std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
  dynamic.pDynamicStates = dynamic_states.data();

  VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.stageCount = frag != VK_NULL_HANDLE ? 2 : 1;
  info.pStages = stages.data();
  info.pVertexInputState = &input;
  info.pInputAssemblyState = &assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth;
  info.pColorBlendState = &color;
  info.pDynamicState = &dynamic;
  info.layout = pipeline_layout_;
  info.renderPass = pass;
  VkPipeline out = VK_NULL_HANDLE;
  check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &out),
        "vkCreateGraphicsPipelines");
  return out;
}

void VulkanRenderer::create_pipelines() {
  struct Modules {
    VkDevice device;
    std::vector<VkShaderModule> all;
    ~Modules() {
      for (VkShaderModule m : all) vkDestroyShaderModule(device, m, nullptr);
    }
  } modules{device_, {}};
  auto load = [&](const uint32_t* code, size_t bytes) {
    modules.all.push_back(shader(code, bytes));
    return modules.all.back();
  };
  const VkShaderModule scene_vert = load(kSceneVert, sizeof(kSceneVert));
  const VkShaderModule scene_frag = load(kSceneFrag, sizeof(kSceneFrag));
  const VkShaderModule shadow_vert = load(kShadowVert, sizeof(kShadowVert));
  const VkShaderModule sky_vert = load(kSkyVert, sizeof(kSkyVert));
  const VkShaderModule sky_frag = load(kSkyFrag, sizeof(kSkyFrag));

  scene_pipeline_ =
      pipeline(scene_vert, scene_frag, main_pass_, true, true, VK_COMPARE_OP_LESS, false);
  shadow_pipeline_ =
      pipeline(shadow_vert, VK_NULL_HANDLE, shadow_pass_, true, true, VK_COMPARE_OP_LESS, true);
  // The sky fills whatever the scene left at the far plane.
  sky_pipeline_ =
      pipeline(sky_vert, sky_frag, main_pass_, false, false, VK_COMPARE_OP_LESS_OR_EQUAL, false);
}

void VulkanRenderer::create_meshes() {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto begin = [&](Mesh& mesh) {
    mesh.first_index = static_cast<uint32_t>(indices.size());
    mesh.vertex_offset = static_cast<int32_t>(vertices.size());
  };
  auto end = [&](Mesh& mesh) {
    mesh.index_count = static_cast<uint32_t>(indices.size()) - mesh.first_index;
  };
  auto grid = [&](uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t a = r * (cols + 1) + c, b = a + cols + 1;
        indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
      }
    }
  };
  constexpr uint32_t kSlices = 48, kStacks = 24;

  // A unit sphere at the origin.
  begin(sphere_);
  for (uint32_t i = 0; i <= kStacks; ++i) {
    const double phi = M_PI * i / kStacks;
    for (uint32_t j = 0; j <= kSlices; ++j) {
      const double theta = 2 * M_PI * j / kSlices;
      const float x = static_cast<float>(std::sin(phi) * std::cos(theta));
      const float y = static_cast<float>(std::cos(phi));
      const float z = static_cast<float>(std::sin(phi) * std::sin(theta));
      vertices.push_back({{x, y, z}, {x, y, z}});
    }
  }
  grid(kStacks, kSlices);
  end(sphere_);

  // An open unit cylinder along +y, from y = 0 to 1 (capsules add end spheres).
  begin(cylinder_);
  for (uint32_t i = 0; i <= 1; ++i) {
    for (uint32_t j = 0; j <= kSlices; ++j) {
      const double theta = 2 * M_PI * j / kSlices;
      const float x = static_cast<float>(std::cos(theta));
      const float z = static_cast<float>(std::sin(theta));
      vertices.push_back({{x, static_cast<float>(i), z}, {x, 0, z}});
    }
  }
  grid(1, kSlices);
  end(cylinder_);

  // The floor: a unit quad in the xz plane.
  begin(floor_);
  for (const float z : {-1.0f, 1.0f}) {
    for (const float x : {-1.0f, 1.0f}) vertices.push_back({{x, 0, z}, {0, 1, 0}});
  }
  grid(1, 1);
  end(floor_);

  vertices_ = make_buffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  std::memcpy(vertices_.mapped, vertices.data(), vertices.size() * sizeof(Vertex));
  indices_ = make_buffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  std::memcpy(indices_.mapped, indices.data(), indices.size() * sizeof(uint32_t));
}

uint32_t VulkanRenderer::memory_type(uint32_t bits,
                                     std::initializer_list<VkMemoryPropertyFlags> choices,
                                     VkMemoryPropertyFlags* chosen) const {
  for (VkMemoryPropertyFlags wanted : choices) {
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & wanted) == wanted) {
        if (chosen) *chosen = memory_.memoryTypes[i].propertyFlags;
        return i;
      }
    }
  }
  throw std::runtime_error("no suitable Vulkan memory type");
}

VulkanRenderer::Buffer VulkanRenderer::make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                                   bool readback) {
  Buffer out;
  out.size = size;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device_, &info, nullptr, &out.buffer), "vkCreateBuffer");
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, out.buffer, &req);
  const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  const VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  VkMemoryPropertyFlags flags = 0;
  // Reading back from uncached (write-combined) memory is slow, so frames
  // land in cached memory when the device has it.
  const uint32_t type =
      readback
          ? memory_type(req.memoryTypeBits,
                        {visible | cached | coherent, visible | cached, visible | coherent}, &flags)
          : memory_type(req.memoryTypeBits, {visible | coherent}, &flags);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = type;
  check(vkAllocateMemory(device_, &alloc, nullptr, &out.memory), "vkAllocateMemory");
  check(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");
  check(vkMapMemory(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped), "vkMapMemory");
  out.coherent = (flags & coherent) != 0;
  return out;
}

void VulkanRenderer::destroy_buffer(Buffer& buffer) {
  if (buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer.buffer, nullptr);
  if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device_, buffer.memory, nullptr);
  buffer = Buffer{};
}

VulkanRenderer::Image VulkanRenderer::make_image(uint32_t width, uint32_t height, VkFormat format,
                                                 VkImageUsageFlags usage,
                                                 VkImageAspectFlags aspect) {
  Image out;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  check(vkCreateImage(device_, &info, nullptr, &out.image), "vkCreateImage");
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(device_, out.image, &req);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex =
      memory_type(req.memoryTypeBits, {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0}, nullptr);
  check(vkAllocateMemory(device_, &alloc, nullptr, &out.memory), "vkAllocateMemory");
  check(vkBindImageMemory(device_, out.image, out.memory, 0), "vkBindImageMemory");
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = out.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {aspect, 0, 1, 0, 1};
  check(vkCreateImageView(device_, &view, nullptr, &out.view), "vkCreateImageView");
  return out;
}

void VulkanRenderer::destroy_image(Image& image) {
  if (image.view != VK_NULL_HANDLE) vkDestroyImageView(device_, image.view, nullptr);
  if (image.image != VK_NULL_HANDLE) vkDestroyImage(device_, image.image, nullptr);
  if (image.memory != VK_NULL_HANDLE) vkFreeMemory(device_, image.memory, nullptr);
  image = Image{};
}

VulkanRenderer::Target& VulkanRenderer::target(int width, int height) {
  const auto key = std::make_pair(width, height);
  auto found = targets_.find(key);
  if (found != targets_.end()) return found->second;
  Target& t = targets_[key];
  const auto w = static_cast<uint32_t>(width), h = static_cast<uint32_t>(height);
  t.color = make_image(w, h, kColorFormat,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);
  t.depth = make_image(w, h, kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
  std::array<VkImageView, 2> views{t.color.view, t.depth.view};
  VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fb.renderPass = main_pass_;
  fb.attachmentCount = static_cast<uint32_t>(views.size());
  fb.pAttachments = views.data();
  fb.width = w;
  fb.height = h;
  fb.layers = 1;
  check(vkCreateFramebuffer(device_, &fb, nullptr, &t.framebuffer), "vkCreateFramebuffer");
  t.readback = make_buffer(static_cast<VkDeviceSize>(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           /*readback=*/true);
  return t;
}

std::vector<uint8_t> VulkanRenderer::render(const scene::Scene& scene, const scene::Camera& camera,
                                            int width, int height) {
  if (width <= 0 || height <= 0) throw std::runtime_error("empty frame size");
  Target& frame_target = target(width, height);

  // Instances: the floor, then spheres (including capsule ends), then the
  // capsules' cylinders.
  std::vector<Instance> list;
  list.push_back(
      instance(basis(Vec3(kFloorExtent, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, kFloorExtent), Vec3()),
               Vec3(), /*floor=*/1));
  const uint32_t first_sphere = static_cast<uint32_t>(list.size());
  auto add_sphere = [&](const Vec3& center, double r, const Vec3& color) {
    list.push_back(instance(basis(Vec3(r, 0, 0), Vec3(0, r, 0), Vec3(0, 0, r), center), color));
  };
  for (const scene::Sphere& s : scene.spheres) add_sphere(s.center, s.radius, s.color);
  for (const scene::Capsule& c : scene.capsules) {
    add_sphere(c.a, c.radius, c.color);
    add_sphere(c.b, c.radius, c.color);
  }
  const uint32_t first_cylinder = static_cast<uint32_t>(list.size());
  for (const scene::Capsule& c : scene.capsules) {
    const Vec3 axis = c.b - c.a;
    const double length = std::sqrt(scene::dot(axis, axis));
    if (length < 1e-9) continue;
    const Vec3 y = axis * (1.0 / length);
    const Vec3 helper = std::abs(y.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    const Vec3 x = scene::normalize(scene::cross(helper, y));
    const Vec3 z = scene::cross(x, y);
    list.push_back(instance(basis(x * c.radius, axis, z * c.radius, c.a), c.color));
  }
  const uint32_t sphere_count = first_cylinder - first_sphere;
  const uint32_t cylinder_count = static_cast<uint32_t>(list.size()) - first_cylinder;
  if (list.size() * sizeof(Instance) > instances_.size) {
    check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
    destroy_buffer(instances_);
    instances_ =
        make_buffer(list.size() * 2 * sizeof(Instance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    write_descriptors();
  }
  std::memcpy(instances_.mapped, list.data(), list.size() * sizeof(Instance));

  // The light's orthographic view covers every caster and the floor patch
  // its shadow falls on.
  const Vec3 light = scene::normalize(scene::light_direction());
  Vec3 lo(1e9, 1e9, 1e9), hi(-1e9, -1e9, -1e9);
  double pad = 0;
  auto extend = [&](const Vec3& p, double r) {
    for (const Vec3& q : {p, p - light * (std::max(p.y, 0.0) / light.y)}) {
      lo = Vec3(std::min(lo.x, q.x), std::min(lo.y, q.y), std::min(lo.z, q.z));
      hi = Vec3(std::max(hi.x, q.x), std::max(hi.y, q.y), std::max(hi.z, q.z));
    }
    pad = std::max(pad, r);
  };
  for (const scene::Sphere& s : scene.spheres) extend(s.center, s.radius);
  for (const scene::Capsule& c : scene.capsules) {
    extend(c.a, c.radius);
    extend(c.b, c.radius);
  }
  if (lo.x > hi.x) lo = hi = Vec3();
  const Vec3 center = (lo + hi) * 0.5;
  const Vec3 diagonal = hi - lo;
  const double radius = 0.5 * std::sqrt(scene::dot(diagonal, diagonal)) + 2 * pad + 0.05;
  const Mat4 light_view_proj = orthographic(radius, 0.01, 2 * radius + 1.0) *
                               look_at(center + light * (radius + 0.5), center, Vec3(0, 1, 0));

  const double aspect = static_cast<double>(width) / height;
  const double half = std::tan(camera.fov_deg * M_PI / 360.0);
  const Vec3 forward = scene::normalize(camera.target - camera.eye);
  const Vec3 right = scene::normalize(scene::cross(forward, Vec3(0, 1, 0)));
  const Vec3 up = scene::cross(right, forward);
  const Mat4 view_proj =
      perspective(camera.fov_deg * M_PI / 180.0, aspect, 0.05, 2 * kFloorExtent) *
      look_at(camera.eye, camera.target, Vec3(0, 1, 0));
  FrameUniforms uniforms{};
  std::memcpy(uniforms.view_proj, view_proj.m.data(), sizeof(uniforms.view_proj));
  std::memcpy(uniforms.light_view_proj, light_view_proj.m.data(), sizeof(uniforms.light_view_proj));
  store(uniforms.eye, camera.eye, 1);
  store(uniforms.light, light, 0);
  store(uniforms.forward, forward, 0);
  store(uniforms.right, right, half * aspect);
  store(uniforms.up, up, half);
  std::memcpy(uniforms_.mapped, &uniforms, sizeof(uniforms));

  // Record: shadow pass, main pass, copy back.
  check(vkResetCommandBuffer(commands_, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(commands_, &begin), "vkBeginCommandBuffer");
  auto set_viewport = [&](uint32_t w, uint32_t h) {
    VkViewport viewport{0, 0, static_cast<float>(w), static_cast<float>(h), 0, 1};
    VkRect2D scissor{{0, 0}, {w, h}};
    vkCmdSetViewport(commands_, 0, 1, &viewport);
    vkCmdSetScissor(commands_, 0, 1, &scissor);
  };
  auto draw = [&](const Mesh& mesh, uint32_t count, uint32_t first) {
    if (count > 0) {
      vkCmdDrawIndexed(commands_, mesh.index_count, count, mesh.first_index, mesh.vertex_offset,
                       first);
    }
  };
  const VkDeviceSize zero = 0;
  vkCmdBindVertexBuffers(commands_, 0, 1, &vertices_.buffer, &zero);
  vkCmdBindIndexBuffer(commands_, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdBindDescriptorSets(commands_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set_,
                          0, nullptr);

  VkClearValue clear_depth{};
  clear_depth.depthStencil = {1.0f, 0};
  VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  pass.renderPass = shadow_pass_;
  pass.framebuffer = shadow_framebuffer_;
  pass.renderArea = {{0, 0}, {kShadowSize, kShadowSize}};
  pass.clearValueCount = 1;
  pass.pClearValues = &clear_depth;
  vkCmdBeginRenderPass(commands_, &pass, VK_SUBPASS_CONTENTS_INLINE);
  set_viewport(kShadowSize, kShadowSize);
  vkCmdBindPipeline(commands_, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
  draw(sphere_, sphere_count, first_sphere);
  draw(cylinder_, cylinder_count, first_cylinder);
  vkCmdEndRenderPass(commands_);

  std::array<VkClearValue, 2> clears{};
  clears[1] = clear_depth;
  pass.renderPass = main_pass_;
  pass.framebuffer = frame_target.framebuffer;
  pass.renderArea = {{0, 0}, {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}};
  pass.clearValueCount = static_cast<uint32_t>(clears.size());
  pass.pClearValues = clears.data();
  vkCmdBeginRenderPass(commands_, &pass, VK_SUBPASS_CONTENTS_INLINE);
  set_viewport(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  vkCmdBindPipeline(commands_, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_);
  draw(floor_, 1, 0);
  draw(sphere_, sphere_count, first_sphere);
  draw(cylinder_, cylinder_count, first_cylinder);
  vkCmdBindPipeline(commands_, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
  vkCmdDraw(commands_, 3, 1, 0, 0);
  vkCmdEndRenderPass(commands_);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  vkCmdCopyImageToBuffer(commands_, frame_target.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         frame_target.readback.buffer, 1, &region);
  VkBufferMemoryBarrier to_host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.buffer = frame_target.readback.buffer;
  to_host.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(commands_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                       nullptr, 1, &to_host, 0, nullptr);
  check(vkEndCommandBuffer(commands_), "vkEndCommandBuffer");

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &commands_;
  check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
  check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  check(vkResetFences(device_, 1, &fence_), "vkResetFences");
  if (!frame_target.readback.coherent) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = frame_target.readback.memory;
    range.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(device_, 1, &range), "vkInvalidateMappedMemoryRanges");
  }

  // RGBA8 -> the env's RGB8.
  const size_t pixels = static_cast<size_t>(width) * height;
  std::vector<uint8_t> rgb(pixels * 3);
  const auto* rgba = static_cast<const uint8_t*>(frame_target.readback.mapped);
  for (size_t i = 0; i < pixels; ++i) {
    rgb[i * 3 + 0] = rgba[i * 4 + 0];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return rgb;
}

void VulkanRenderer::destroy() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    for (auto& [size, t] : targets_) {
      if (t.framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device_, t.framebuffer, nullptr);
      destroy_image(t.color);
      destroy_image(t.depth);
      destroy_buffer(t.readback);
    }
    targets_.clear();
    destroy_buffer(vertices_);
    destroy_buffer(indices_);
    destroy_buffer(uniforms_);
    destroy_buffer(instances_);
    for (VkPipeline* p : {&scene_pipeline_, &shadow_pipeline_, &sky_pipeline_}) {
      if (*p != VK_NULL_HANDLE) vkDestroyPipeline(device_, *p, nullptr);
      *p = VK_NULL_HANDLE;
    }
    if (descriptor_pool_ != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    if (shadow_sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, shadow_sampler_, nullptr);
    if (shadow_framebuffer_ != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device_, shadow_framebuffer_, nullptr);
    destroy_image(shadow_map_);
    if (main_pass_ != VK_NULL_HANDLE) vkDestroyRenderPass(device_, main_pass_, nullptr);
    if (shadow_pass_ != VK_NULL_HANDLE) vkDestroyRenderPass(device_, shadow_pass_, nullptr);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;
}

}  // namespace

std::unique_ptr<Renderer> make_vulkan_renderer() { return std::make_unique<VulkanRenderer>(); }

}  // namespace chrono_reach
