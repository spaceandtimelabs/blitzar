#pragma once

#include <optional>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// basic_copier
//--------------------------------------------------------------------------------------------------
class basic_copier {
 public:

   xena::future<> copy(basct::cspan<std::byte> src) noexcept;

 private:
};
} // sxt::xendv
