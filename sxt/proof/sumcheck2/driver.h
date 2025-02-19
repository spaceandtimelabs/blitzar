#pragma once

#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/base/field/element.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/proof/sumcheck2/workspace.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class driver {
public:
  virtual ~driver() noexcept = default;

  virtual xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles,
                 basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept = 0;
  
  virtual xena::future<> sum(basct::span<T> polynomial,
                             workspace& ws) const noexcept = 0;
  
  virtual xena::future<> fold(workspace& ws, const T& r) const noexcept = 0;
};
} // namespace sxt::prfsk2
