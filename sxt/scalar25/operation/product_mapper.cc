#include "sxt/scalar25/operation/product_mapper.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// async_make_host_mapper
//--------------------------------------------------------------------------------------------------
product_mapper product_mapper::async_make_host_mapper(void* host_data, bast::raw_stream_t stream,
                                                      unsigned int n,
                                                      unsigned int offset) const noexcept {
  SXT_DEBUG_ASSERT(offset < n);
  auto half_num_bytes = (n - offset) * sizeof(s25t::element);
  basdv::async_memcpy_device_to_host(host_data, lhs_data_ + offset, half_num_bytes, stream);
  void* host_data_rhs = static_cast<char*>(host_data) + half_num_bytes;
  basdv::async_memcpy_device_to_host(host_data_rhs, rhs_data_ + offset, half_num_bytes, stream);
  return product_mapper{
      reinterpret_cast<const s25t::element*>(host_data),
      reinterpret_cast<const s25t::element*>(host_data_rhs),
  };
}
} // namespace sxt::s25o
