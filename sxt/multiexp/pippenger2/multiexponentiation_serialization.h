#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "sxt/multiexp/pippenger2/partition_table_accessor.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/container/span.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// variable_length_multiexponentiation_descriptor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
struct variable_length_multiexponentiation_descriptor {
  std::unique_ptr<partition_table_accessor<U>> accessor;
  std::vector<unsigned> output_bit_table;
  std::vector<unsigned> output_lengths;
  std::vector<uint8_t> scalars;
};

//--------------------------------------------------------------------------------------------------
// write_multiexponentiation 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
void write_multiexponentiation(const char* dir, const partition_table_accessor<U>& accessor,
                               basct::cspan<unsigned> output_bit_table,
                               basct::cspan<unsigned> output_lengths,
                               basct::cspan<uint8_t> scalars) noexcept;
} // namespace sxt::mtxpp2
