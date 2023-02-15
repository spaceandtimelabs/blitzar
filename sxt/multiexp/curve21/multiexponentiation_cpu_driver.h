#pragma once

#include "sxt/multiexp/pippenger/driver2.h"

namespace sxt::mtxc21 {
class multiproduct_solver;

//--------------------------------------------------------------------------------------------------
// multiexponentiation_cpu_driver
//--------------------------------------------------------------------------------------------------
class multiexponentiation_cpu_driver final : public mtxpi::driver2 {
public:
  explicit multiexponentiation_cpu_driver(const multiproduct_solver* solver) noexcept;

  // mtxpi::driver2
  xena::future<memmg::managed_array<void>>
  compute_multiproduct(mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
                       const basct::blob_array& masks, size_t num_inputs) const noexcept override;

  xena::future<memmg::managed_array<void>>
  combine_multiproduct_outputs(xena::future<memmg::managed_array<void>>&& multiproduct,
                               basct::blob_array&& output_digit_or_all) const noexcept override;

private:
  const multiproduct_solver* solver_;
};
} // namespace sxt::mtxc21
