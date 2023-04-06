#include "sxt/scalar25/operation/inner_product.h"

#include <algorithm>

#include "sxt/algorithm/reduction/reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/scalar25/operation/accumulator.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/product_mapper.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inner_product
//--------------------------------------------------------------------------------------------------
void inner_product(s25t::element& res, basct::cspan<s25t::element> lhs,
                   basct::cspan<s25t::element> rhs) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  SXT_DEBUG_ASSERT(n > 0);
  s25o::mul(res, lhs[0], rhs[0]);
  for (size_t i = 1; i < n; ++i) {
    s25o::muladd(res, lhs[i], rhs[i], res);
  }
}

//--------------------------------------------------------------------------------------------------
// async_inner_product
//--------------------------------------------------------------------------------------------------
xena::future<s25t::element> async_inner_product(basct::cspan<s25t::element> lhs,
                                                basct::cspan<s25t::element> rhs) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  SXT_DEBUG_ASSERT(n > 0);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<s25t::element> device_data{&resource};
  size_t buffer_size = 0;
  auto is_device_lhs = basdv::is_device_pointer(lhs.data());
  auto is_device_rhs = basdv::is_device_pointer(rhs.data());
  buffer_size = (static_cast<size_t>(!is_device_lhs) + static_cast<size_t>(!is_device_rhs)) * n;
  if (buffer_size > 0) {
    device_data = memmg::managed_array<s25t::element>{buffer_size, &resource};
  }
  auto data = device_data.data();
  if (!is_device_lhs) {
    basdv::async_copy_host_to_device(basct::span<s25t::element>{data, n}, lhs.subspan(0, n),
                                     stream);
    lhs = {data, n};
    data += n;
  }
  if (!is_device_rhs) {
    basdv::async_copy_host_to_device(basct::span<s25t::element>{data, n}, rhs.subspan(0, n),
                                     stream);
    rhs = {data, n};
  }
  return algr::reduce<accumulator>(std::move(stream), product_mapper{lhs.data(), rhs.data()},
                                   static_cast<unsigned int>(n));
}
} // namespace sxt::s25o
