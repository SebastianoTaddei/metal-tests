#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tensor.hpp"

namespace backend
{

struct Tensor::Impl
{
  std::vector<float> data;

  Impl(std::vector<float> const &data) : data(data) {}
};

Tensor::Tensor(std::vector<float> const &data) : impl(std::make_unique<Impl>(data)) {}

Tensor::~Tensor() {}

void Tensor::to(Device const *device)
{
  assert(device != nullptr);
  assert(device->type() == Type::METAL);

  this->device = device;
}

std::vector<float> Tensor::cpu() const { return this->impl->data; }

} // namespace backend
