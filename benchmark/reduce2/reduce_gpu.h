#pragma once

namespace sxt::c21t { struct element_p3; }

namespace sxt {
//--------------------------------------------------------------------------------------------------
// reduce_gpu
//--------------------------------------------------------------------------------------------------
void reduce_gpu(c21t::element_p3* res, int m, int n) noexcept;
} // namespace sxt

