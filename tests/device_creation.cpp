#include <iostream>

#include "device.hpp"

int main()
{
  auto cpu_device   = gpu_playground::make_cpu_device();
  auto metal_device = gpu_playground::make_metal_device();

  std::cout << "The CPU device is: " << gpu_playground::get_device_name(cpu_device->type()) << '\n';
  std::cout << "The Metal device is: " << gpu_playground::get_device_name(metal_device->type()) << '\n';
  return 0;
}
