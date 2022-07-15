#pragma once

#include <cstdint>
#include <type_traits>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(rstt::compressed_element& r, basct::cspan<uint8_t> a,
                     const rstt::compressed_element& p) noexcept;

template <class T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>* = nullptr>
void scalar_multiply(rstt::compressed_element& h, T a, const rstt::compressed_element& p) noexcept {
  scalar_multiply(h, basct::cspan<uint8_t>{reinterpret_cast<uint8_t*>(&a), sizeof(a)}, p);
}
} // namespace sxt::rsto
