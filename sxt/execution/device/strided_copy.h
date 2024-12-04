#pragma once

#include <cstddef>

#include "sxt/execution/async/future_fwd.h"

namespace sxt::basdv { class stream; }

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy 
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy(std::byte* dst, const basdv::stream& stream, const std::byte* src,
                            size_t n, size_t count, size_t stride) noexcept;
} // namespace sxt::xendv
