#pragma once

#include "buffer.hpp"
#include "device.hpp"

namespace gpu_playground
{

class Tensor
{
private:
  std::shared_ptr<Device> device;
  Buffer buffer;

public:
  Tensor(std::vector<float> data, DevicePtr const &device)
      : device(device), buffer(this->device->new_buffer(std::move(data)))
  {
  }

  Tensor(Tensor const &other)
      : device(other.device), buffer(this->device->new_buffer_with_size(other.buffer.size))
  {
    this->device->copy_buffer(other.buffer, this->buffer);
  }

  Tensor &operator=(Tensor const &other)
  {
    this->device = other.device;
    this->device->copy_buffer(other.buffer, this->buffer);

    return *this;
  }

  Tensor &operator+=(Tensor const &rhs)
  {
    backend::assert_compatible(this->buffer, rhs.buffer);

    this->device->add(this->buffer, rhs.buffer, this->buffer);

    return *this;
  }

  friend Tensor operator+(Tensor lhs, Tensor const &rhs);

  std::vector<float> cpu() { return this->device->cpu(this->buffer); }
};

inline Tensor operator+(Tensor lhs, Tensor const &rhs)
{
  lhs += rhs;
  return lhs;
}

} // namespace gpu_playground
