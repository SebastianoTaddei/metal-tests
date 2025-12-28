#pragma once

#include "device.hpp"

namespace backend
{

class CPUDevice final : public Device
{
public:
  CPUDevice();

  ~CPUDevice();

  Type type() const override { return Type::CPU; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  Buffer new_buffer(std::vector<float> data) const override;

  std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace backend
