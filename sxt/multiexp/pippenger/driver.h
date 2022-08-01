#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxb {
class exponent;
}
namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
public:
  virtual ~driver() noexcept = default;

  virtual void compute_multiproduct_inputs(memmg::managed_array<void>& inout,
                                           basct::cspan<basct::cspan<size_t>> powers,
                                           size_t radix_log2, size_t num_multiproduct_inputs,
                                           size_t num_multiproduct_entries) const noexcept = 0;

  virtual void compute_multiproduct(memmg::managed_array<void>& inout,
                                    mtxi::index_table& multiproduct_table,
                                    size_t num_inputs) const noexcept = 0;

  virtual void
  combine_multiproduct_outputs(memmg::managed_array<void>& inout,
                               basct::cspan<uint8_t> output_digit_or_all) const noexcept = 0;
};
} // namespace sxt::mtxpi
