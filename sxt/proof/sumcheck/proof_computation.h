#pragma once

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"

namespace sxt::prft { class transcript; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// prove_sum 
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials, prft::transcript& transcript,
                         basct::cspan<s25t::element> mles,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept;
} // namespace sxt::prfsk
