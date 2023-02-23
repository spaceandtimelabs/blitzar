#pragma once

#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// multiexponentiation_fn
//--------------------------------------------------------------------------------------------------
using multiexponentiation_fn = basf::function_ref<memmg::managed_array<c21t::element_p3>(
    basct::cspan<c21t::element_p3>, basct::cspan<mtxb::exponent_sequence> exponents)>;

//--------------------------------------------------------------------------------------------------
// exercise_multiexponentiation_fn
//--------------------------------------------------------------------------------------------------
void exercise_multiexponentiation_fn(std::mt19937& rng, multiexponentiation_fn f) noexcept;
} // namespace sxt::mtxtst
