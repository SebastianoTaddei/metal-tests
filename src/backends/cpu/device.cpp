#include <string>

#include "device.hpp"

namespace backend
{

struct Device::Impl
{
  std::string const name{"CPU"};
};

Device::Device() : impl(new Impl()) {}

Device::~Device() { delete this->impl; }

std::string const &Device::name() const & { return this->impl->name; }

} // namespace backend
