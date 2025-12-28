#include "Eigen/Dense"

#include "cpu_device.hpp"

namespace backend
{

using CPUBuffer = Eigen::VectorXf;

CPUDevice::CPUDevice() {}

CPUDevice::~CPUDevice() {}

void CPUDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  auto cpu_a = static_cast<CPUBuffer const *>(a.handle.get());
  auto cpu_b = static_cast<CPUBuffer const *>(b.handle.get());
  auto cpu_c = static_cast<CPUBuffer *>(c.handle.get());
  assert(cpu_a->size() == cpu_b->size());
  assert(cpu_b->size() == cpu_c->size());

  *cpu_c = *cpu_a + *cpu_b;
}

Buffer CPUDevice::new_buffer(std::vector<float> data) const
{
  return Buffer{
    .handle =
      HandlePtr{
        new CPUBuffer(Eigen::Map<CPUBuffer>(data.data(), data.size())),
        [](void *ptr) -> void { delete static_cast<CPUBuffer *>(ptr); }
      },
    .size = data.size(),
  };
}

std::vector<float> CPUDevice::cpu(Buffer const &buffer) const
{
  auto cpu_buffer = static_cast<CPUBuffer const *>(buffer.handle.get());
  return std::vector<float>(cpu_buffer->data(), cpu_buffer->data() + cpu_buffer->size());
}

} // namespace backend
