#include "sxt/execution/device/strided_copy.h"

#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> strided_copy(void* dst, const basdv::stream& stream, const void* src,
                            size_t num_bytes, size_t stride, size_t offset) noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::xendv
