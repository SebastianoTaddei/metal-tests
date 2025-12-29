#include <iostream>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

int main()
{
  constexpr size_t len{3};

  auto cpu_device = gpu_playground::make_cpu_device();
  auto mtl_device = gpu_playground::make_metal_device();

  std::vector<float> const a{0.0, 1.0, 2.0};
  std::vector<float> const b{0.0, 1.0, 2.0};
  std::vector<float> c(len, 0.0);

  auto const cpu_a = gpu_playground::Tensor(a, cpu_device);
  auto const cpu_b = gpu_playground::Tensor(b, cpu_device);
  auto cpu_c       = gpu_playground::Tensor(c, cpu_device);
  auto const mtl_a = gpu_playground::Tensor(a, mtl_device);
  auto const mtl_b = gpu_playground::Tensor(b, mtl_device);
  auto mtl_c       = gpu_playground::Tensor(c, mtl_device);

  cpu_c = cpu_a + cpu_b;
  mtl_c = mtl_a + mtl_b;

  c = cpu_c.cpu();
  std::cout << "CPU add: ";
  for (size_t i{0}; i < len; i++)
  {
    std::cout << c.at(i) << " ";
  }
  std::cout << '\n';

  c = mtl_c.cpu();
  std::cout << "Metal add: ";
  for (size_t i{0}; i < len; i++)
  {
    std::cout << c.at(i) << " ";
  }
  std::cout << '\n';

  return 0;
}
