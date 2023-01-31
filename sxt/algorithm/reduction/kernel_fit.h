#pragma once

namespace sxt::xenk {
struct kernel_dims;
}

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// fit_reduction_kernel
//--------------------------------------------------------------------------------------------------
xenk::kernel_dims fit_reduction_kernel(unsigned int n) noexcept;
} // namespace sxt::algr
