#pragma once

namespace sxt::c21t {
struct element_p3;
}

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multi_exp_gpu
//--------------------------------------------------------------------------------------------------
void multi_exp_gpu(c21t::element_p3* res, int m, int n) noexcept;
} // namespace sxt
