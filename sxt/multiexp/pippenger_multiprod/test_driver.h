#pragma once

#include "sxt/multiexp/pippenger_multiprod/driver.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// test_driver
//--------------------------------------------------------------------------------------------------
class test_driver final : public driver {
public:
  void apply_partition_operation(basct::span_void inout, basct::cspan<uint64_t> partition_markers,
                                 size_t partition_size) const noexcept override;

  void apply_clump2_operation(basct::span_void inout, basct::cspan<uint64_t> markers,
                              const mtxi::clump2_descriptor& descriptor) const noexcept override;

  void compute_naive_multiproduct(basct::span_void inout,
                                  basct::cspan<basct::cspan<uint64_t>> products,
                                  size_t num_inactive_inputs) const noexcept override;

  void permute_inputs(basct::span_void inout,
                      basct::cspan<uint64_t> permutation) const noexcept override;

private:
};
} // namespace sxt::mtxpmp
