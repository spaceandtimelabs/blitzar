#pragma once

namespace sxt::c21t { struct element_p3; }

namespace sxt::c21p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
bool is_on_curve(const c21t::element_p3& p) noexcept;
} // namespace sxt::c21p
