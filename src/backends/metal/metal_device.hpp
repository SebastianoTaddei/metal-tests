#pragma once

#include "device.hpp"

namespace backend
{

class MetalDevice final : public Device
{
private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;

public:
  MetalDevice();

  ~MetalDevice();

  Type type() const override { return Type::METAL; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  Buffer new_buffer(std::vector<float> data) const override;

  std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace backend
