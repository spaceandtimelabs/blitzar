#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::sqcb { class commitment; }

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_function
//--------------------------------------------------------------------------------------------------
void test_pedersen_get_generators(
    basf::function_ref<void(basct::span<sqcb::commitment> generators,
        uint64_t offset_generators)> f
);
}  // namespace sxt::sqctst
