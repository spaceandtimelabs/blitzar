#pragma once

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::prft { class transcript; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
class driver;

//--------------------------------------------------------------------------------------------------
// prove_sum 
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials,
                         basct::span<s25t::element> evaluation_point, prft::transcript& transcript,
                         const driver& drv, basct::cspan<s25t::element> mles,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept;
} // namespace sxt::prfsk
