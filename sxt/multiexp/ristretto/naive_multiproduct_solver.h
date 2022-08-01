#pragma once

#include "sxt/multiexp/ristretto/multiproduct_solver.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// naive_multiproduct_solver
//--------------------------------------------------------------------------------------------------
class naive_multiproduct_solver final : public multiproduct_solver {
public:
  size_t workspace_size(size_t num_multiproduct_inputs, size_t num_entries) const noexcept override;

  void solve(memmg::managed_array<void>& inout, mtxi::index_table& multiproduct_table,
             size_t num_inputs) const noexcept override;
};
} // namespace sxt::mtxrs
