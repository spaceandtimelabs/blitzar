#pragma once

namespace sxt::f51t {
class element;
}

namespace sxt::c21b {
//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
void apply_elligator(f51t::element& x, f51t::element& y, int* notsquare_p,
                     const f51t::element& r) noexcept;
} // namespace sxt::c21b
