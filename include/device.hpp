#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

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
  Device();
  ~Device();

  Type type() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace backend
