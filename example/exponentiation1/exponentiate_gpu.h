#pragma once

namespace sxt::c21t { struct element_p3; }

namespace sxt {
//--------------------------------------------------------------------------------------------------
// exponentiate_gpu
//--------------------------------------------------------------------------------------------------
void exponentiate_gpu(c21t::element_p3* res, int n) noexcept;
}  // namespace sxt
