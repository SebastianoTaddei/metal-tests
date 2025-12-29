#include "Eigen/Dense"
#include "buffer.hpp"

#include "cpu_device.hpp"

namespace gpu_playground::backend
{

using CPUBuffer = Eigen::VectorXf;

CPUDevice::CPUDevice() {}

CPUDevice::~CPUDevice() {}

void CPUDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible(a, b, c);

  auto cpu_a = static_cast<CPUBuffer const *>(a.handle.get());
  auto cpu_b = static_cast<CPUBuffer const *>(b.handle.get());
  auto cpu_c = static_cast<CPUBuffer *>(c.handle.get());

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
    .size        = data.size(),
    .device_type = CPUDevice::s_type,
  };
}

Buffer CPUDevice::new_buffer_with_size(size_t size) const
{
  auto const data = std::vector<float>(size, 0.0);
  return this->new_buffer(std::move(data));
}

void CPUDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible(from, to);

  auto cpu_from = static_cast<CPUBuffer const *>(from.handle.get());
  auto cpu_to   = static_cast<CPUBuffer *>(to.handle.get());

  *cpu_to = *cpu_from;
}

std::vector<float> CPUDevice::cpu(Buffer const &buffer) const
{
  auto cpu_buffer = static_cast<CPUBuffer const *>(buffer.handle.get());
  return std::vector<float>(cpu_buffer->data(), cpu_buffer->data() + cpu_buffer->size());
}

std::unique_ptr<Device> make_cpu_device();

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_cpu_device()
{
  return std::make_shared<gpu_playground::backend::CPUDevice>();
}
