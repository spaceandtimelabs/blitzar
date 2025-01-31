#include "sxt/base/system/directory_recorder.h"

#include <cstdlib>
#include <filesystem>
#include <format>

#include "sxt/base/error/panic.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// try_get_directory
//--------------------------------------------------------------------------------------------------
static std::string_view try_get_directory() noexcept {
  static std::string res = [] {
    auto val = std::getenv("BLITZAR_DUMP_DIR");
    if (val != nullptr) {
      return std::string{val};
    } else {
      return std::string{};
    }
  }();
  return res;
}

//--------------------------------------------------------------------------------------------------
// counter
//--------------------------------------------------------------------------------------------------
std::atomic<unsigned> directory_recorder::counter_{0};

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
directory_recorder::directory_recorder(std::string base_name,
                                       std::string_view force_record_dir) noexcept {
  std::string_view dir;
  if(!force_record_dir.empty()) {
    dir = force_record_dir;
  } else {
    dir = try_get_directory();
  }
  if (dir.empty()) {
    return;
  }
  auto i = counter_++;
  name_ = std::format("{}/{}-{}", dir, base_name, i);
  try {
    std::filesystem::create_directory(name_);
  } catch(const std::exception& e) {
    baser::panic("failed to create directory {}: {}", name_, e.what());
  }
}
} // namespace sxt::bassy
