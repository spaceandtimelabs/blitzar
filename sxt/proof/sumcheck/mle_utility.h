#pragma once

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basdv { class stream; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles 
//--------------------------------------------------------------------------------------------------
void copy_partial_mles(memmg::managed_array<s25t::element>& partial_mles, basdv::stream& stream,
                       basct::cspan<s25t::element> mles, unsigned n, unsigned a,
                       unsigned b) noexcept;
} // namespace sxt::prfsk
