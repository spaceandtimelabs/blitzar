#pragma once

#include <cstddef>

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_components
//--------------------------------------------------------------------------------------------------
void init_precomputed_components(size_t n, bool use_gpu) noexcept;
} // namespace sxt::sqcgn
