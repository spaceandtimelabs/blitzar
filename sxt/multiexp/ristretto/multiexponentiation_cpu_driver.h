#pragma once

#include <cstddef>
#include <cstring>
#include <iostream>

#include "sxt/multiexp/pippenger/driver.h"

namespace sxt::mtxrs {
class input_accessor;
class multiproduct_solver;

//--------------------------------------------------------------------------------------------------
// multiexponentiation_cpu_driver
//--------------------------------------------------------------------------------------------------
class multiexponentiation_cpu_driver final : public sxt::mtxpi::driver {
public:
  explicit multiexponentiation_cpu_driver(
      const mtxrs::input_accessor* input_accessor = nullptr,
      const mtxrs::multiproduct_solver* multiproduct_solver = nullptr) noexcept;

  void compute_multiproduct_inputs(memmg::managed_array<void>& inout,
                                   basct::cspan<basct::cspan<size_t>> powers, size_t radix_log2,
                                   size_t num_multiproduct_inputs,
                                   size_t num_multiproduct_entries) const noexcept override;

  void compute_multiproduct(memmg::managed_array<void>& inout,
                            mtxi::index_table& multiproduct_table,
                            size_t num_inputs) const noexcept override;

  virtual void
  combine_multiproduct_outputs(memmg::managed_array<void>& inout,
                               basct::cspan<uint8_t> output_digit_or_all) const noexcept override;

private:
  const input_accessor* input_accessor_;
  const multiproduct_solver* multiproduct_solver_;
};
} // namespace sxt::mtxrs
