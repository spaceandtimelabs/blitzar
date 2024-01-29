#pragma once

#include <string_view>
#include <format>

#include "sxt/base/log/log_impl.h"

namespace sxt::basl {
//--------------------------------------------------------------------------------------------------
// info
//--------------------------------------------------------------------------------------------------
template <class... Args>
void info(std::format_string<Args...> fmt, Args&&... args) noexcept {
  info_impl(std::format(fmt, std::forward<Args>(args)...));
}
} // namespace sxt::basl
