#pragma once

#include <cstddef>
#include <vector>

#include "sxt/multiexp/base/exponent.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// exponent_aggregates_counts
//--------------------------------------------------------------------------------------------------
struct exponent_aggregates {
  std::vector<mtxb::exponent> term_or_all;
  std::vector<mtxb::exponent> output_or_all;
  mtxb::exponent max_exponent;
  size_t pop_count;
};
} // namespace sxt::mtxpi
