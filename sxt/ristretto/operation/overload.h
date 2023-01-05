#pragma once

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
class element;
}

namespace sxt::rstt {
class compressed_element;

//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
compressed_element operator+(const compressed_element& lhs, const compressed_element& rhs) noexcept;
compressed_element operator+(const c21t::element_p3& lhs, const compressed_element& rhs) noexcept;
compressed_element operator+(const compressed_element& lhs, const c21t::element_p3& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
compressed_element operator-(const compressed_element& lhs, const compressed_element& rhs) noexcept;
compressed_element operator-(const compressed_element& x) noexcept;
compressed_element operator-(const c21t::element_p3& lhs, const compressed_element& rhs) noexcept;
compressed_element operator-(const compressed_element& lhs, const c21t::element_p3& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
compressed_element operator*(const s25t::element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
compressed_element& operator+=(compressed_element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
compressed_element& operator-=(compressed_element& lhs, const compressed_element& rhs) noexcept;
} // namespace sxt::rstt
