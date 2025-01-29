#pragma once

#include <cstddef>

#include "sxt/base/curve/element.h"
#include "sxt/base/container/span.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// write_multiexponentiation 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void write_multiexponentiation(const char* dir, basct::cspan<unsigned> output_bit_table,
                               basct::cspan<unsigned> output_lengths,
                               basct::cspan<uint8_t> scalars) noexcept;
/* template <bascrv::element T, class U> */
/*   requires std::constructible_from<T, U> */
/* void multiexponentiate(basct::span<T> res, const partition_table_accessor<U>& accessor, */
/*                        basct::cspan<unsigned> output_bit_table, */
/*                        basct::cspan<unsigned> output_lengths, */
/*                        basct::cspan<uint8_t> scalars) noexcept { */
} // namespace sxt::mtxpp2
