#include "sxt/base/log/log_impl.h"

#include "spdlog/spdlog.h"
#include "sxt/base/log/setup.h"

namespace sxt::basl {
//--------------------------------------------------------------------------------------------------
// info_impl
//--------------------------------------------------------------------------------------------------
void info_impl(std::string_view s) noexcept {
  setup_logger();
  spdlog::info(s);
}
} // namespace sxt::basl
