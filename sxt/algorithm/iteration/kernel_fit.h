#pragma once

namespace sxt::xenk {
struct kernel_dims;
}

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// fit_iteration_kernel
//--------------------------------------------------------------------------------------------------
xenk::kernel_dims fit_iteration_kernel(unsigned int n) noexcept;
} // namespace sxt::algi
