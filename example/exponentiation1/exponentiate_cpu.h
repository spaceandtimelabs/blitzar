#pragma once

namespace sxt::c21t { struct element_p3; }

namespace sxt {
//--------------------------------------------------------------------------------------------------
// exponentiate_cpu
//--------------------------------------------------------------------------------------------------
void exponentiate_cpu(c21t::element_p3* res, int n) noexcept;
}  // namespace sxt

