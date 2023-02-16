#pragma once

#include <concepts>

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
concept mapper = requires(M m, typename M::value_type& x, unsigned int i, void* data) {
  { m.map_index(i) } noexcept -> std::convertible_to<typename M::value_type>;
  { m.map_index(x, i) } noexcept;
};
} // namespace sxt::algb
