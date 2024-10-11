#pragma once

#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/proof/sumcheck/workspace.h"

namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
  public:
    virtual ~driver() noexcept = default;

    virtual xena::future<std::unique_ptr<workspace>>
    make_workspace(basct::cspan<s25t::element> mles,
                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                   basct::cspan<unsigned> product_terms, unsigned n) const noexcept = 0;

    virtual xena::future<> sum(basct::span<s25t::element> polynomial,
                               workspace& ws) const noexcept = 0;

    virtual xena::future<> fold(workspace& ws, const s25t::element& r) const noexcept = 0;
};
} // namespace sxt::prfsk
