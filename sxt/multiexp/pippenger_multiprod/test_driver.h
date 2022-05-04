#pragma once

#include "sxt/multiexp/pippenger_multiprod/driver.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// test_driver
//--------------------------------------------------------------------------------------------------
class test_driver final : public driver {
 public:
  void apply_partition_operation(memmg::managed_array<void>& inout,
                                 basct::cspan<uint64_t> partition_markers,
                                 size_t partition_size) const noexcept override;

   void apply_clump2_operation(
       memmg::managed_array<void>& inout, basct::cspan<uint64_t> markers,
       const mtxi::clump2_descriptor& descriptor) const noexcept override;
 private:
};
} // namespace sxt::mtxpmp
