#pragma once

#include "sxt/base/container/span.h"

namespace sxt::rstt {
class compressed_element;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// compress
//--------------------------------------------------------------------------------------------------
void compress(rstt::compressed_element& e_p, const c21t::element_p3& e) noexcept;

//--------------------------------------------------------------------------------------------------
// decompress
//--------------------------------------------------------------------------------------------------
void decompress(c21t::element_p3& e_p, const rstt::compressed_element& e) noexcept;

//--------------------------------------------------------------------------------------------------
// batch_compress
//--------------------------------------------------------------------------------------------------
void batch_compress(basct::span<rstt::compressed_element> ex_p,
                    basct::cspan<c21t::element_p3> ex) noexcept;
} // namespace sxt::rsto
