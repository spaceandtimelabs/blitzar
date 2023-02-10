#pragma once

#include "sxt/multiexp/curve21/multiproduct_solver.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// pippenger_multiproduct_solver
//--------------------------------------------------------------------------------------------------
class pippenger_multiproduct_solver final : public multiproduct_solver {
public:
  // multiproduct_solver
  xena::future<memmg::managed_array<c21t::element_p3>>
  solve(mtxi::index_table&& multiproduct_table, basct::cspan<c21t::element_p3> generators,
        const basct::blob_array& masks, size_t num_inputs) const noexcept override;
};
} // namespace sxt::mtxc21
