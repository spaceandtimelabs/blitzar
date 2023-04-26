#pragma once

namespace sxt::f51t {
class element;
}

namespace sxt::c21b {
//--------------------------------------------------------------------------------------------------
// mont_to_ed
//--------------------------------------------------------------------------------------------------
void mont_to_ed(f51t::element& xed, f51t::element& yed, const f51t::element& x,
                const f51t::element& y) noexcept;
} // namespace sxt::c21b
