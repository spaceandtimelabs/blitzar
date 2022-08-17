#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sxt/base/container/blob_array.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// exponent_aggregates_counts
//--------------------------------------------------------------------------------------------------
struct exponent_aggregates {
  basct::blob_array term_or_all;
  basct::blob_array output_or_all;
  std::vector<uint8_t> max_exponent;
  size_t pop_count;
};
} // namespace sxt::mtxpi
