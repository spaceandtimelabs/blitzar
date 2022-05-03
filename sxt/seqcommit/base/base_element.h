#pragma once

#include <cstdint>

namespace sxt::c21t { struct element_p3; }

namespace sxt::sqcb {
//--------------------------------------------------------------------------------------------------
// compute_base_element
//--------------------------------------------------------------------------------------------------
void compute_base_element(c21t::element_p3& g, uint64_t index) noexcept;
} // namespace sxt::sqcb
