#pragma once

#include <memory>
#include <vector>

#include "device.hpp"

namespace backend
{

class Tensor
{
public:
  Tensor(std::vector<float> const &data);
  ~Tensor();

  void to(Device const *device);
  std::vector<float> cpu() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  Device const *device{nullptr};
};

} // namespace backend
