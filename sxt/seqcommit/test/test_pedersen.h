#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::sqcb { class commitment; }
namespace sxt::mtxb { struct exponent_sequence; }

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_function
//--------------------------------------------------------------------------------------------------
void test_pedersen_compute_commitment(
    basf::function_ref<void(
        basct::span<sqcb::commitment>,
        basct::cspan<mtxb::exponent_sequence>,
        basct::cspan<sqcb::commitment> generators
    )> f
);

}  // namespace sxt::sqctst
