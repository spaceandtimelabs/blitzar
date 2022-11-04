#pragma once

#include <string>
#include <type_traits>

#include "sxt/base/container/span.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/proof/transcript/transcript_primitive.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// append
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<is_transcript_primitive_v<T>>* = nullptr>
inline void append(transcript& trans, std::string_view label, const T& value) {
  trans.append_message(label, {reinterpret_cast<const uint8_t*>(&value), sizeof(T)});
}

template <class T, std::enable_if_t<is_transcript_primitive_v<T>>* = nullptr>
inline void append(transcript& trans, std::string_view label, basct::cspan<T> values) {
  trans.append_message(
      label, {reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T)});
}
} // namespace sxt::prft
