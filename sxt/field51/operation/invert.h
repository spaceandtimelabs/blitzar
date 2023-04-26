#pragma once

namespace sxt::f51t {
class element;
}

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// invert
//--------------------------------------------------------------------------------------------------
/*
 * Inversion - returns 0 if z=0
 */
void invert(f51t::element& out, const f51t::element& z) noexcept;
} // namespace sxt::f51o
