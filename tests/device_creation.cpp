#include <iostream>

#include "cpu_device.hpp"
#include "metal_device.hpp"

int main()
{
  auto cpu_device   = backend::CPUDevice();
  auto metal_device = backend::MetalDevice();

  std::cout << "The CPU device is: " << backend::get_device_name(cpu_device.type()) << '\n';
  std::cout << "The Metal device is: " << backend::get_device_name(metal_device.type()) << '\n';
  return 0;
}
