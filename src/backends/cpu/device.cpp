#include "device.hpp"

namespace backend
{

struct Device::Impl
{
  static constexpr Type type{Type::CPU};
};

Device::Device() : impl(std::make_unique<Impl>()) {}

Device::~Device() {}

Type Device::type() const { return Device::Impl::type; }

} // namespace backend
