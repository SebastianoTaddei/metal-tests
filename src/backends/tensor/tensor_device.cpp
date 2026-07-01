#include <memory>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/utils/xutils.hpp>

#include "tensor_device.hpp"

namespace gpu_playground::backend
{

using TENSORBuffer = xt::xarray<float, xt::layout_type::row_major>;

namespace
{

struct Add
{
  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, TENSORBuffer const &b) const
  {
    return a + b;
  }

  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, float const b) const
  {
    return a + b;
  }
};

struct Sub
{
  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, TENSORBuffer const &b) const
  {
    return a - b;
  }

  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, float const b) const
  {
    return a - b;
  }
};

struct Mul
{
  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, TENSORBuffer const &b) const
  {
    return a * b;
  }

  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, float const b) const
  {
    return a * b;
  }
};

struct Div
{
  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, TENSORBuffer const &b) const
  {
    return a / b;
  }

  [[nodiscard]] TENSORBuffer operator()(TENSORBuffer const &a, float const b) const
  {
    return a / b;
  }
};

template <class Op>
void cwisem_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_same_shape(a, b, c);

  auto const &tensor_a = *static_cast<TENSORBuffer const *>(a.get());
  auto const &tensor_b = *static_cast<TENSORBuffer const *>(b.get());
  auto &tensor_c       = *static_cast<TENSORBuffer *>(c.get());

  tensor_c = op(tensor_a, tensor_b);
}

template <class Op>
void cwises_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_compatible_sop(a, b, c);

  auto const &tensor_a = *static_cast<TENSORBuffer const *>(a.get());
  auto const &tensor_b = *static_cast<TENSORBuffer const *>(b.get());
  auto &tensor_c       = *static_cast<TENSORBuffer *>(c.get());

  auto const scalar_b = tensor_b(0);
  tensor_c            = op(tensor_a, scalar_b);
}

} // namespace

void TENSORDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Add{});
}

void TENSORDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Sub{});
}

void TENSORDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_mul(a, b, c);

  auto const &tensor_a = *static_cast<TENSORBuffer const *>(a.get());
  auto const &tensor_b = *static_cast<TENSORBuffer const *>(b.get());
  auto &tensor_c       = *static_cast<TENSORBuffer *>(c.get());

  tensor_c = tensor_a * tensor_b;
}

void TENSORDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Mul{});
}

void TENSORDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Div{});
}

void TENSORDevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Add{});
}

void TENSORDevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Sub{});
}

void TENSORDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Mul{});
}

void TENSORDevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Div{});
}

Buffer TENSORDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  return Buffer{
      HandlePtr{
          new TENSORBuffer(
              xt::xarray_adaptor<std::vector<float>, xt::layout_type::row_major>(
                  data, {shape.rows, shape.cols}
              )
          ),
          [](void *ptr) -> void
          { std::default_delete<TENSORBuffer>{}(static_cast<TENSORBuffer *>(ptr)); }
      },
      shape,
      TENSORDevice::s_type
  };
}

void TENSORDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible_copy(from, to);

  auto const &tensor_from = *static_cast<TENSORBuffer const *>(from.get());
  auto &tensor_to         = *static_cast<TENSORBuffer *>(to.get());

  tensor_to = tensor_from;
}

void TENSORDevice::transpose(Buffer const &from, Buffer &to) const
{
  assert_compatible_transpose(from, to);

  auto const &tensor_from = *static_cast<TENSORBuffer const *>(from.get());
  auto &tensor_to         = *static_cast<TENSORBuffer *>(to.get());

  tensor_to = xt::transpose(tensor_from);
}

std::vector<float> TENSORDevice::cpu(Buffer const &buffer) const
{
  auto const &tensor_buffer = *static_cast<TENSORBuffer const *>(buffer.get());
  return {tensor_buffer.cbegin(), tensor_buffer.cend()};
}

void TENSORDevice::sync([[maybe_unused]] Buffer const &buffer) const {}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_tensor_device()
{
  return std::make_shared<gpu_playground::backend::TENSORDevice>();
}
