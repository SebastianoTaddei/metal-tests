#pragma once

#include <memory>
#include <vector>

#include "buffer.hpp"
#include "device_types.hpp"

namespace gpu_playground
{

class Device
{
public:
  virtual ~Device() = default;

  virtual DeviceType type() const                                     = 0;
  virtual void add(Buffer const &a, Buffer const &b, Buffer &c) const = 0;
  virtual Buffer new_buffer(std::vector<float> data) const            = 0;
  virtual Buffer new_buffer_with_size(size_t size) const              = 0;
  virtual void copy_buffer(Buffer const &from, Buffer &to) const      = 0;
  virtual std::vector<float> cpu(Buffer const &buffer) const          = 0;
};

using DevicePtr = std::shared_ptr<Device>;

DevicePtr make_cpu_device();

#ifdef GPU_PLAYGROUND_HAS_METAL
DevicePtr make_metal_device();
#endif

} // namespace gpu_playground
