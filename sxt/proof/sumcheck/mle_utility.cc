#include "sxt/proof/sumcheck/mle_utility.h"

#include "sxt/base/device/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
void copy_partial_mles(memmg::managed_array<s25t::element> partial_mles, basdv::stream& stream,
                       basct::cspan<s25t::element> mles, unsigned n, unsigned a,
                       unsigned b) noexcept {}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
