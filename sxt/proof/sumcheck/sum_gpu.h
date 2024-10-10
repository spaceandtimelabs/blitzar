#pragma once

#include "sxt/base/container/span.h"

namespace sxt::basdv { class stream; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
void sum(basct::span<s25t::element> polynomial, basdv::stream& stream,
         basct::cspan<s25t::element> mles,
         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
         basct::cspan<unsigned> product_terms, unsigned mid, unsigned n) noexcept;
} // namespace sxt::prfsk
