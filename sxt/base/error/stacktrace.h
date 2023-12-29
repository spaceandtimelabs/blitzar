#pragma once

#include <string>

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// stacktrace
//--------------------------------------------------------------------------------------------------
/**
 * Wrapper around an implementation of stacktrace.
 *
 * In the future, we might replace this with https://en.cppreference.com/w/cpp/header/stacktrace
 * once it's available in standard libraries
 */
std::string stacktrace() noexcept;
} // namespace sxt::baser
