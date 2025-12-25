#pragma once
#include <string>

namespace backend
{

class Device
{
public:
  Device();
  ~Device();

  std::string const &name() const &;

private:
  struct Impl;
  Impl *impl;
};

} // namespace backend
