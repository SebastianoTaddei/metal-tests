#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class EigenDevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::EIGEN};

public:
  EigenDevice();

  ~EigenDevice();

  DeviceType type() const override { return EigenDevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  Buffer new_buffer(std::vector<float> data) const override;

  Buffer new_buffer_with_size(size_t size) const override;

  void copy_buffer(Buffer const &from, Buffer &to) const override;

  std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
