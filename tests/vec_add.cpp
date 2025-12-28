#include <iostream>
#include <vector>

#include "cpu_device.hpp"
#include "metal_device.hpp"

int main()
{
  constexpr size_t len{3};

  auto cpu_device = backend::CPUDevice();
  auto mtl_device = backend::MetalDevice();

  std::vector<float> const a{0.0, 1.0, 2.0};
  std::vector<float> const b{0.0, 1.0, 2.0};
  std::vector<float> c(len, 0.0);

  auto const cpu_a = cpu_device.new_buffer(a);
  auto const cpu_b = cpu_device.new_buffer(b);
  auto cpu_c       = cpu_device.new_buffer(c);
  auto const mtl_a = mtl_device.new_buffer(a);
  auto const mtl_b = mtl_device.new_buffer(b);
  auto mtl_c       = mtl_device.new_buffer(c);

  cpu_device.add(cpu_a, cpu_b, cpu_c);
  mtl_device.add(mtl_a, mtl_b, mtl_c);

  c = cpu_device.cpu(cpu_c);
  std::cout << "CPU add: ";
  for (size_t i{0}; i < len; i++)
  {
    std::cout << c.at(i) << " ";
  }
  std::cout << '\n';

  c = mtl_device.cpu(mtl_c);
  std::cout << "Metal add: ";
  for (size_t i{0}; i < len; i++)
  {
    std::cout << c.at(i) << " ";
  }
  std::cout << '\n';

  return 0;
}
