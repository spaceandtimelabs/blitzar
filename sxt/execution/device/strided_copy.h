#pragma once

#include "sxt/execution/async/future_fwd.h"

namespace sxt::basdv { class stream; }

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy 
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy(void* dst, const basdv::stream& stream, const void* src,
                            size_t num_bytes, size_t stride, size_t offset) noexcept;
} // namespace sxt::xendv
