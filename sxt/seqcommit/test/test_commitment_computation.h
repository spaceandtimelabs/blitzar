#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::sqcb { class commitment; }
namespace sxt::mtxb { struct exponent_sequence; }

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_commitment_computation_function
//--------------------------------------------------------------------------------------------------
void test_commitment_computation_function(
    basf::function_ref<void(basct::span<sqcb::commitment>,
                            basct::cspan<mtxb::exponent_sequence>)>
        f);
}  // namespace sxt::sqctst
