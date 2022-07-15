#pragma once

namespace sxt::c21t {
struct element_p3;
}

namespace sxt {
//--------------------------------------------------------------------------------------------------
// reduce_cpu
//--------------------------------------------------------------------------------------------------
void reduce_cpu(c21t::element_p3* res, int m, int n) noexcept;
} // namespace sxt
