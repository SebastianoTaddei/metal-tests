#include <iostream>

#include "device.hpp"

int main()
{
  backend::Device device{};

  std::cout << "The device is: " << device.name() << '\n';
  return 0;
}
