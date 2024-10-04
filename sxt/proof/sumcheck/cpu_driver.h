#pragma once

#include "sxt/proof/sumcheck/driver.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// cpu_driver
//--------------------------------------------------------------------------------------------------
class cpu_driver final : public driver {

  // driver
  std::unique_ptr<workspace>
  make_workspace(basct::cspan<s25t::element> mles,
                 basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override;

  xena::future<> sum(basct::span<s25t::element> polynomial,
                             workspace& ws) const noexcept override;

  xena::future<> fold(workspace& ws, const s25t::element& r) const noexcept override;
};
} // namespace prfsk
