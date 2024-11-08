#pragma once

#include "sxt/base/container/span.h"

namespace sxt::basdv { class stream; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reduce_sums 
//--------------------------------------------------------------------------------------------------
void reduce_sums(basct::span<s25t::element> p, basdv::stream& stream,
                 basct::cspan<s25t::element> partial_terms, unsigned num_terms) noexcept;
} // namespace sxt::prfsk
