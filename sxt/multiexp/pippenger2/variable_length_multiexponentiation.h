#pragma once

#include "sxt/multiexp/pippenger2/multiexponentiation.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// async_multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_multiexponentiate(basct::span<T> res,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<unsigned> output_bit_table,
                                       basct::cspan<unsigned> output_lengths,
                                       basct::cspan<uint8_t> scalars) noexcept {
  multiexponentiate_options options;
  options.split_factor = static_cast<unsigned>(basdv::get_num_devices());
  return multiexponentiate_impl(res, accessor, output_bit_table, output_lengths, scalars, options);
}
} // namespace sxt::mtxpp2
