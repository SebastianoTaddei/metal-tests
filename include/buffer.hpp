#pragma once

#include <functional>
#include <memory>

#include "device_types.hpp"

namespace gpu_playground
{

using HandlePtr = std::unique_ptr<void, std::function<void(void *)>>;

struct Buffer
{
  HandlePtr handle;
  size_t size;
  Type type;
};

} // namespace gpu_playground
