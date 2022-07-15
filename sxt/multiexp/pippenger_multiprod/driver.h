#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct {
class span_void;
}
namespace sxt::mtxi {
struct clump2_descriptor;
}

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
public:
  virtual ~driver() noexcept = default;

  virtual void apply_partition_operation(basct::span_void inout,
                                         basct::cspan<uint64_t> partition_markers,
                                         size_t partition_size) const noexcept = 0;

  virtual void apply_clump2_operation(basct::span_void inout, basct::cspan<uint64_t> markers,
                                      const mtxi::clump2_descriptor& descriptor) const noexcept = 0;

  virtual void compute_naive_multiproduct(basct::span_void inout,
                                          basct::cspan<basct::cspan<uint64_t>> products,
                                          size_t num_inactive_inputs) const noexcept = 0;

  void compute_naive_multiproduct(basct::span_void inout,
                                  basct::span<basct::span<uint64_t>> products,
                                  size_t num_inactive_inputs) const noexcept;

  virtual void permute_inputs(basct::span_void inout,
                              basct::cspan<uint64_t> permutation) const noexcept = 0;
};
} // namespace sxt::mtxpmp
