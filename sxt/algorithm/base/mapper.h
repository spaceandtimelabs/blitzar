#pragma once

#include <concepts>
#include <cstddef>

#include "sxt/base/type/raw_stream.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// mapper
//--------------------------------------------------------------------------------------------------
/**
 * Describe a generic map function that can be used within CUDA kernels.
 *
 * Mapper turns an index into a value.
 */
template <class M>
concept mapper = requires(M m, typename M::value_type& x, unsigned int i, bast::raw_stream_t stream,
                          void* data) {
  { m.map_index(i) } noexcept -> std::convertible_to<typename M::value_type>;
  { m.map_index(x, i) } noexcept;
  { m.async_make_host_mapper(data, stream, i, i) } noexcept -> std::convertible_to<M>;
  { M::num_bytes_per_index } -> std::convertible_to<size_t>;
};
} // namespace sxt::algb
