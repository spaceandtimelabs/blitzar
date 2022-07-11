#pragma once

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t { struct element_p3; }
namespace sxt::rstt { struct compressed_element; }

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// compute_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_base_element(c21t::element_p3& g, uint64_t index) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_compressed_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_compressed_base_element(rstt::compressed_element& g_rt, uint64_t index) noexcept;
} // namespace sxt::sqcgn
