#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class CPUDevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::CPU};

public:
  CPUDevice();

  ~CPUDevice();

  DeviceType type() const override { return CPUDevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  Buffer new_buffer(std::vector<float> data) const override;

  std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
