#pragma once

#include <cstddef>

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// reduction_stats
//--------------------------------------------------------------------------------------------------
struct reduction_stats {
  size_t prev_num_terms;
  size_t num_terms;
};
} // namespace sxt::mtxpmp

