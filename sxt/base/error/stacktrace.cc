#include "sxt/base/error/stacktrace.h"

#define BOOST_STACKTRACE_USE_BACKTRACE
#include "boost/stacktrace.hpp"

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
std::string stacktrace() noexcept {
  return boost::stacktrace::to_string(boost::stacktrace::stacktrace());
}
} // namespace sxt::baser
