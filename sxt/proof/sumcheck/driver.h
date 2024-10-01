#pragma once

#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/proof/sumcheck/workspace.h"

namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
  public:
    virtual ~driver() noexcept = default;

    virtual std::unique_ptr<workspace>
    make_workspace(basct::cspan<s25t::element> mles,
                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                   basct::cspan<unsigned> product_terms) const noexcept = 0;
};
} // namespace sxt::prfsk
