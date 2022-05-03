#pragma once

#include <cstdint>

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// clump2_descriptor
//--------------------------------------------------------------------------------------------------
struct clump2_descriptor {
  uint64_t size;
  uint64_t subset_count;  // the number of subsets of {1, .., size} of
                          // at most cardinality 2
};
}  // namespace sxt::mtxi
