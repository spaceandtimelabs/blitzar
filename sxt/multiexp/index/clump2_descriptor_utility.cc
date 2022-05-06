#include "sxt/multiexp/index/clump2_descriptor_utility.h"

#include "sxt/multiexp/index/clump2_descriptor.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// init_clump2_descriptor
//--------------------------------------------------------------------------------------------------
void init_clump2_descriptor(clump2_descriptor& descriptor,
                            uint64_t clump_size) noexcept {
  descriptor.size = clump_size;
  descriptor.subset_count = clump_size * (clump_size - 1) / 2 + clump_size;
}
}  // namespace sxt::mtxi
