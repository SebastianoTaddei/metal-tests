#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device.hpp"

namespace backend
{

struct Device::Impl
{
  static constexpr Type type{Type::METAL};

  id<MTLDevice> device{nil};

  Impl()
  {
    this->device = MTLCreateSystemDefaultDevice();
    assert(this->device != nil);
  }

  ~Impl()
  {
    // ARC handles device lifetime
    this->device = nil;
  }

  bool valid() const { return this->device != nil; }
};

Device::Device() : impl(std::make_unique<Impl>()) {}

Device::~Device() {}

Type Device::type() const { return Device::Impl::type; }

} // namespace backend
