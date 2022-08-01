#pragma once

#include <cstddef>

#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// multiproduct_solver
//--------------------------------------------------------------------------------------------------
class multiproduct_solver {
public:
  virtual ~multiproduct_solver() noexcept = default;

  virtual size_t workspace_size(size_t num_multiproduct_inputs,
                                size_t num_entries) const noexcept = 0;

  virtual void solve(memmg::managed_array<void>& inout, mtxi::index_table& multiproduct_table,
                     size_t num_inputs) const noexcept = 0;
};
} // namespace sxt::mtxrs
