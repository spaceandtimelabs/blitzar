#include "sxt/base/system/directory_recorder.h"

#include <fstream>
#include <format>

#include "sxt/base/test/temp_directory.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::bassy;

TEST_CASE("we can set up a directory for recording") {
  bastst::temp_directory dir;
  directory_recorder recorder{"abc", dir.name()};
  std::ofstream out{std::format("{}/t", recorder.name())};
  REQUIRE(out.good());
}
