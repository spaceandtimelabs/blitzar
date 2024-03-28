#pragma once

#include "sxt/base/curve/element.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// generator_accessor 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
class generator_accessor {
 public:
   virtual ~generator_accessor() noexcept = default;

};
} // namespace sxt::mtxb
