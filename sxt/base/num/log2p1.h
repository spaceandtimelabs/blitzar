#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// log2p1
//--------------------------------------------------------------------------------------------------
/** 
* - x represents an unsigned number in little-endian format
* - x can have a maximum of 1016 bits
* - x.size() represents the total amount of bytes for the number
*/
double log2p1(basct::cspan<uint8_t> x) noexcept;

} // namesape sxt::basn
