#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

#include "device.hpp"

namespace backend
{

struct Device::Impl
{
  std::string const name{"Metal"};

  id<MTLDevice> device = nil;

  Impl() { device = MTLCreateSystemDefaultDevice(); }

  ~Impl()
  {
    // ARC handles device lifetime
    device = nil;
  }

  bool valid() const { return device != nil; }
};

Device::Device() : impl(new Impl()) {}

Device::~Device() { delete this->impl; }

std::string const &Device::name() const & { return this->impl->name; }

} // namespace backend
