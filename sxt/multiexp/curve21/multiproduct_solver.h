#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::mtxi {
class index_table;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// multiproduct_solver
//--------------------------------------------------------------------------------------------------
class multiproduct_solver {
public:
  virtual ~multiproduct_solver() noexcept = default;

  virtual xena::future<memmg::managed_array<c21t::element_p3>>
  solve(mtxi::index_table&& multiproduct_table, basct::cspan<c21t::element_p3> generators,
        const basct::blob_array& mask, size_t num_inputs) const noexcept = 0;
};
} // namespace sxt::mtxc21
