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
  virtual std::vector<float> cpu(Buffer const &buffer) const          = 0;
};

std::unique_ptr<Device> make_cpu_device();

#ifdef GPU_PLAYGROUND_HAS_METAL
std::unique_ptr<Device> make_metal_device();
#endif

namespace backend
{

template <typename First, typename... Rest>
inline void assert_same_device(First const &first, Rest const &...rest)
{
  const DeviceType ref = first.type;
  ((assert(rest.type == ref && "Buffers are on different devices")), ...);
}

template <typename First, typename... Rest>
inline void assert_same_size(First const &first, Rest const &...rest)
{
  const std::size_t ref = first.size;
  ((assert(rest.size == ref && "Buffers have different sizes")), ...);
}

template <typename First, typename... Rest>
inline void assert_compatible(First const &first, Rest const &...rest)
{
  assert_same_device(first, rest...);
  assert_same_size(first, rest...);
}

} // namespace backend

} // namespace gpu_playground
