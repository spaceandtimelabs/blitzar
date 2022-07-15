#pragma once

#include <cstdint>

#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::sqcb {
//--------------------------------------------------------------------------------------------------
// indexed_exponent_sequence
//--------------------------------------------------------------------------------------------------
struct indexed_exponent_sequence {
  mtxb::exponent_sequence exponent_sequence;

  // when `indices` is nullptr, then the sequence descriptor
  // represents a dense sequence. In this case, data[i] is
  // always tied with row i.
  // If `indices` is not nullptr, then the sequence represents
  // a sparse sequence such that `indices[i]` holds
  // the actual row_i in which data[i] is tied with. In case
  // indices is not nullptr, then `indices` must have
  // exactly `n` elements.
  const uint64_t* indices = nullptr;
};
} // namespace sxt::sqcb
