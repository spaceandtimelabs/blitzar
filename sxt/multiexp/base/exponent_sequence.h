#pragma once

#include <cstdint>

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// exponent_sequence
//--------------------------------------------------------------------------------------------------
struct exponent_sequence {
  // the number of bytes used to represent an element in the sequence
  // element_nbytes must be a power of 2 and must satisfy
  //    1 <= element_nbytes <= 32
  uint8_t element_nbytes;

  // the number of elements in the sequence
  uint64_t n;

  // pointer to the data for the sequence of elements where there are n elements
  // in the sequence and each element enocodes a number of element_nbytes bytes
  // represented in the little endian format
  const uint8_t* data;
};
} // namespace sxt::mtxb
