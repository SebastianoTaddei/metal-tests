#include <iostream>

#include "device.hpp"

int main()
{
  backend::Device device{};

  std::cout << "The device is: " << backend::get_device_name(device.type()) << '\n';
  return 0;
}
