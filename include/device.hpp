#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "buffer.hpp"

namespace backend
{

#define DEVICE_TYPES                                                                               \
  X(CPU)                                                                                           \
  X(METAL)

enum class Type : uint8_t
{
#define X(type) type,
  DEVICE_TYPES
#undef X
    COUNT
};

inline constexpr std::array<std::string_view, static_cast<size_t>(Type::COUNT)> device_names{
#define X(name) #name,
  DEVICE_TYPES
#undef X
};

inline constexpr std::string_view get_device_name(Type const type)
{
  return device_names.at(static_cast<size_t>(type));
}

class Device
{
public:
  virtual ~Device() = default;

  virtual Type type() const                                           = 0;
  virtual void add(Buffer const &a, Buffer const &b, Buffer &c) const = 0;
  virtual Buffer new_buffer(std::vector<float> data) const            = 0;
  virtual std::vector<float> cpu(Buffer const &buffer) const          = 0;
};

} // namespace backend
