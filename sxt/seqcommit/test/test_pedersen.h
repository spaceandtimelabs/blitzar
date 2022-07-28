#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::rstt {
class compressed_element;
}
namespace sxt::sqcb {
struct indexed_exponent_sequence;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_function
//--------------------------------------------------------------------------------------------------
void test_pedersen_compute_commitment(
    basf::function_ref<void(basct::span<rstt::compressed_element>,
                            basct::cspan<sqcb::indexed_exponent_sequence>,
                            basct::cspan<c21t::element_p3> generators)>
        f);

} // namespace sxt::sqctst
