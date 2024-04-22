#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/property.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// combine 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine(basct::span<T> res, bast::raw_stream_t stream, basct::cspan<T> elements) noexcept {
  (void)res;
  (void)stream;
  (void)elements;
}
} // namespace sxt::mtxpp2
