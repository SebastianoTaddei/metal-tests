#pragma once

#include <functional>
#include <memory>

namespace backend
{

using HandlePtr = std::unique_ptr<void, std::function<void(void *)>>;

struct Buffer
{
  HandlePtr handle;
  size_t size;
};

} // namespace backend
