#pragma once

#include <type_traits>

namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::s25t {
struct element;
}

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// is_transcript_primitive_v
//--------------------------------------------------------------------------------------------------
template <class T>
constexpr bool is_transcript_primitive_v =
    std::is_integral_v<T> || std::is_same_v<T, unsigned char> ||
    std::is_same_v<T, rstt::compressed_element> || std::is_same_v<T, s25t::element>;
} // namespace sxt::prft
