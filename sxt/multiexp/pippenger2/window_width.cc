#include "sxt/multiexp/pippenger2/window_width.h"

#include <cstdlib>
#include <cstring>
#include <charconv>

#include "sxt/base/error/panic.h"
#include "sxt/base/log/log.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// get_default_window_width_impl 
//--------------------------------------------------------------------------------------------------
static unsigned get_default_window_width_impl() noexcept {
  auto s = std::getenv("BLITZAR_PARTITION_WINDW_WIDTH");
  if (s == nullptr) {
    return 16;
  }
  unsigned width;
  auto parse_result = std::from_chars(s, s + std::strlen(s), width);
  if (parse_result.ec != std::errc{}) {
    baser::panic("failed to parse partition window width {}", s);
  }
  if (width == 0) {
    baser::panic("partition window width cannot be zero");
  }
  return width;
}

//--------------------------------------------------------------------------------------------------
// get_default_window_width
//--------------------------------------------------------------------------------------------------
unsigned get_default_window_width() noexcept {
  static auto res = []() noexcept {
    auto res = get_default_window_width_impl();
    basl::info("using a default partition window width of {}", res);
    return res;
  }();
  return res;
}
} // sxt::mtxpp2
