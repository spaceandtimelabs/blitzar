#pragma once

#include <cstdint>

namespace sxt::c21t {
struct element_p3;

//--------------------------------------------------------------------------------------------------
// form_point
//--------------------------------------------------------------------------------------------------
void form_point(element_p3& p, const uint8_t r[32]) noexcept;
} // namespace sxt::c21t
