#pragma once

namespace sxt::f51t {
class element;
}

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// sqmul
//--------------------------------------------------------------------------------------------------
void sqmul(f51t::element& s, int n, const f51t::element& a) noexcept;
} // namespace sxt::f51o
