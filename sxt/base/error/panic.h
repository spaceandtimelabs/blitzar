#pragma once

#include <string_view>

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// panic
//--------------------------------------------------------------------------------------------------
/**
 * Note: This technique for getting the file and line is technically not standard compliant, but
 * it works with most compilers (gcc, clang) and the standard-friendly approach requires c++20
 * and support is flaky:
 * https://en.cppreference.com/w/cpp/utility/source_location
 */
[[noreturn]] void panic(std::string_view message, int line = __builtin_LINE(),
                        const char* file = __builtin_FILE()) noexcept;
} // namespace sxt::baser
