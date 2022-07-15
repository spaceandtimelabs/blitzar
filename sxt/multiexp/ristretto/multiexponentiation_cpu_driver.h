#pragma once

#include <cstddef>
#include <cstring>
#include <iostream>

#include "sxt/multiexp/pippenger/driver.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// multiexponentiation_cpu_driver
//--------------------------------------------------------------------------------------------------
class multiexponentiation_cpu_driver final : public sxt::mtxpi::driver {
public:
  void compute_multiproduct_inputs(memmg::managed_array<void>& inout,
                                   basct::cspan<basct::cspan<size_t>> powers,
                                   size_t radix_log2) const noexcept override;

  void compute_multiproduct(memmg::managed_array<void>& inout,
                            mtxi::index_table& multiproduct_table) const noexcept override;

  virtual void
  combine_multiproduct_outputs(memmg::managed_array<void>& inout,
                               basct::cspan<uint8_t> output_digit_or_all) const noexcept override;
};
} // namespace sxt::mtxrs
